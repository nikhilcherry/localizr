"""localizr: pixel-level transit localization via TESS/Kepler difference imaging.

Public API: :func:`localize` and :class:`LocalizationResult`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time

from . import diffimage, plots
from . import gaia as gaia_module
from . import verdict as verdict_module
from .tpf import TpfResolutionError, resolve_tpf
from .verdict import CENTROID_OFFSET_SIGMA_THRESHOLD

__version__ = "0.1.0"

GAIA_SEARCH_MARGIN_ARCSEC = 30.0

__all__ = [
    "localize",
    "LocalizationResult",
    "LocalizationError",
    "TpfResolutionError",
    "CENTROID_OFFSET_SIGMA_THRESHOLD",
]


class LocalizationError(Exception):
    """Raised when localization fails outright (bad inputs, no TPF, etc.)."""


@dataclass
class LocalizationResult:
    tic_id: int | None
    kic_id: int | None
    mission: str
    centroid_offset_arcsec: float | None
    centroid_offset_sigma: float | None
    target_position: dict[str, float]
    difference_image_centroid: dict[str, float | None]
    nearby_gaia_sources: list[dict[str, Any]]
    verdict: str
    verdict_reason: str
    _diff_image: np.ndarray | None = field(repr=False, default=None)
    _wcs: Any = field(repr=False, default=None)

    def save_plot(self, path: str) -> str:
        """Save the difference-image diagnostic figure (see :mod:`localizr.plots`)."""
        if self._diff_image is None or self._wcs is None:
            raise LocalizationError(
                "no difference image was computed (verdict is "
                f"'{self.verdict}'); nothing to plot"
            )
        label = self.tic_id if self.mission == "TESS" else self.kic_id
        id_kind = "TIC" if self.mission == "TESS" else "KIC"
        return plots.save_plot(
            path,
            self._diff_image,
            self._wcs,
            self.target_position["ra"],
            self.target_position["dec"],
            self.difference_image_centroid["ra"],
            self.difference_image_centroid["dec"],
            self.nearby_gaia_sources,
            title=f"{self.mission} {id_kind} {label} -- {self.verdict}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tic_id": self.tic_id,
            "kic_id": self.kic_id,
            "mission": self.mission,
            "centroid_offset_arcsec": self.centroid_offset_arcsec,
            "centroid_offset_sigma": self.centroid_offset_sigma,
            "target_position": self.target_position,
            "difference_image_centroid": self.difference_image_centroid,
            "nearby_gaia_sources": self.nearby_gaia_sources,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
        }

    def to_json(self, path: str | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text)
        return text


def _footprint_radius_arcsec(flux_shape: tuple[int, int], wcs) -> float:
    ny, nx = flux_shape
    scale = verdict_module.pixel_scale_arcsec(wcs)
    return scale * float(np.hypot(ny, nx)) / 2.0


def localize(
    *,
    tic_id: int | str | None = None,
    kic_id: int | str | None = None,
    tpf_path: str | None = None,
    period: float,
    epoch_btjd: float,
    duration_hours: float,
    sector: int | None = None,
    quarter: int | None = None,
    cache_dir: str | None = None,
    gaia_search_margin_arcsec: float = GAIA_SEARCH_MARGIN_ARCSEC,
) -> LocalizationResult:
    """Localize a transit signal against its target pixel file.

    Exactly one of ``tic_id`` (TESS), ``kic_id`` (Kepler), or ``tpf_path``
    (a local FITS file, any mission) must be given.

    ``epoch_btjd`` must be in the same time system as the TPF's own TIME
    column: BTJD for TESS, BKJD for Kepler (i.e. whatever ``koi_time0bk`` /
    ``pl_tranmid - 2457000`` etc. already gives you for that mission). The
    parameter name follows the TESS convention since that's the primary
    target of this tool, but the value is interpreted generically.

    ``sector``/``quarter`` select a specific TESS sector or Kepler quarter;
    leaving both as ``None`` lets lightkurve pick the first available one
    for the target (any one that covers a repeating ``period``/``epoch_btjd``
    transit works, since the same ephemeris recurs throughout the mission).
    """
    tpf, mission, catalog_id = resolve_tpf(
        tic_id=tic_id, kic_id=kic_id, tpf_path=tpf_path,
        sector=sector, quarter=quarter, cache_dir=cache_dir,
    )

    result_tic_id = catalog_id if mission == "TESS" else None
    result_kic_id = catalog_id if mission == "Kepler" else None

    target_ra, target_dec = float(tpf.ra), float(tpf.dec)
    target_position_raw = {"ra": target_ra, "dec": target_dec}

    time = np.asarray(tpf.time.value, dtype=float)
    flux = tpf.flux.value if hasattr(tpf.flux, "value") else np.asarray(tpf.flux)
    quality = np.asarray(tpf.quality) if getattr(tpf, "quality", None) is not None else None

    try:
        diff_result = diffimage.compute_difference_image(
            flux, time, period, epoch_btjd, duration_hours, quality=quality
        )
    except diffimage.DifferenceImageError as exc:
        return LocalizationResult(
            tic_id=result_tic_id,
            kic_id=result_kic_id,
            mission=mission,
            centroid_offset_arcsec=None,
            centroid_offset_sigma=None,
            target_position=target_position_raw,
            difference_image_centroid={"ra": None, "dec": None},
            nearby_gaia_sources=[],
            verdict=verdict_module.VERDICT_INCONCLUSIVE,
            verdict_reason=str(exc),
        )

    wcs = tpf.wcs
    centroid_ra, centroid_dec = wcs.all_pix2world(
        diff_result.centroid_col, diff_result.centroid_row, 0
    )
    centroid_ra, centroid_dec = float(centroid_ra), float(centroid_dec)

    radius_arcsec = _footprint_radius_arcsec(diff_result.diff_image.shape, wcs) + gaia_search_margin_arcsec
    gaia_error_note = None
    try:
        gaia_sources = gaia_module.query_nearby_sources(target_ra, target_dec, radius_arcsec)
    except gaia_module.GaiaQueryError as exc:
        gaia_sources = []
        gaia_error_note = str(exc)

    obstime = Time(
        float(np.mean(time[diff_result.in_transit_mask])),
        format=tpf.time.format,
        scale=tpf.time.scale,
    )

    vresult = verdict_module.compute_verdict(
        catalog_ra=target_ra,
        catalog_dec=target_dec,
        centroid_ra=centroid_ra,
        centroid_dec=centroid_dec,
        centroid_uncertainty_col=diff_result.centroid_uncertainty_col,
        centroid_uncertainty_row=diff_result.centroid_uncertainty_row,
        wcs=wcs,
        obstime=obstime,
        gaia_sources=gaia_sources,
    )

    reason = vresult.verdict_reason
    if gaia_error_note:
        reason += f" (Gaia cross-match unavailable: {gaia_error_note})"

    sigma = vresult.centroid_offset_sigma
    sigma = sigma if np.isfinite(sigma) else None

    return LocalizationResult(
        tic_id=result_tic_id,
        kic_id=result_kic_id,
        mission=mission,
        centroid_offset_arcsec=vresult.centroid_offset_arcsec,
        centroid_offset_sigma=sigma,
        target_position=vresult.target_position,
        difference_image_centroid={"ra": centroid_ra, "dec": centroid_dec},
        nearby_gaia_sources=[s.to_dict() for s in gaia_sources],
        verdict=vresult.verdict,
        verdict_reason=reason,
        _diff_image=diff_result.diff_image,
        _wcs=wcs,
    )
