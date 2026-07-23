"""Centroid-offset-sigma verdict: on-target, off-target blend, or inconclusive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales

from .gaia import GaiaSource

CENTROID_OFFSET_SIGMA_THRESHOLD = 3.0
TARGET_GAIA_MATCH_RADIUS_ARCSEC = 3.0
# TIC/KIC RA_OBJ/DEC_OBJ are quoted at equinox J2000 regardless of when the
# TPF was actually observed (confirmed via each mission's own FITS header:
# EQUINOX = 2000.0). A raw, un-propagated Gaia DR3 position (ref_epoch 2016.0)
# can be several arcsec off from that for a high-proper-motion star, which is
# enough to miss the match entirely -- so candidates are propagated back to
# J2000 for the *matching* step, then to the real observation epoch afterward
# for the actual centroid comparison.
CATALOG_REFERENCE_EPOCH = Time(2000.0, format="jyear")

VERDICT_ON_TARGET = "on_target"
VERDICT_OFF_TARGET_BLEND = "off_target_blend"
VERDICT_INCONCLUSIVE = "inconclusive"


@dataclass
class VerdictResult:
    target_position: dict
    centroid_offset_arcsec: float
    centroid_offset_sigma: float
    verdict: str
    verdict_reason: str


def pixel_scale_arcsec(wcs) -> float:
    """Average pixel scale of ``wcs`` in arcsec/pixel."""
    scales_deg = proj_plane_pixel_scales(wcs)
    return float(np.mean(scales_deg)) * 3600.0


def match_target_gaia_source(
    catalog_ra: float,
    catalog_dec: float,
    gaia_sources: list[GaiaSource],
    max_sep_arcsec: float = TARGET_GAIA_MATCH_RADIUS_ARCSEC,
    catalog_epoch: Time = CATALOG_REFERENCE_EPOCH,
) -> GaiaSource | None:
    """Nearest Gaia source to the target's nominal catalog position, if any is close enough.

    Candidates are proper-motion-propagated to ``catalog_epoch`` (J2000 by
    default) before comparing, since (ra, dec) is normally an equinox-J2000
    catalog position rather than one already corrected to the observation
    epoch -- see :data:`CATALOG_REFERENCE_EPOCH`.
    """
    if not gaia_sources:
        return None
    target = SkyCoord(ra=catalog_ra * u.deg, dec=catalog_dec * u.deg)
    best, best_sep = None, None
    for source in gaia_sources:
        sep = target.separation(propagate_position(source, catalog_epoch)).arcsec
        if best_sep is None or sep < best_sep:
            best, best_sep = source, sep
    if best_sep is not None and best_sep <= max_sep_arcsec:
        return best
    return None


def propagate_position(source: GaiaSource, obstime: Time) -> SkyCoord:
    """Propagate a Gaia source's position from its reference epoch to ``obstime``."""
    distance = None
    if source.parallax is not None and source.parallax > 0:
        distance = (1000.0 / source.parallax) * u.pc

    coord = SkyCoord(
        ra=source.ra * u.deg,
        dec=source.dec * u.deg,
        pm_ra_cosdec=(source.pmra or 0.0) * u.mas / u.yr,
        pm_dec=(source.pmdec or 0.0) * u.mas / u.yr,
        distance=distance,
        obstime=Time(source.ref_epoch, format="jyear"),
        frame="icrs",
    )
    return coord.apply_space_motion(new_obstime=obstime)


def compute_verdict(
    *,
    catalog_ra: float,
    catalog_dec: float,
    centroid_ra: float,
    centroid_dec: float,
    centroid_uncertainty_col: float,
    centroid_uncertainty_row: float,
    wcs,
    obstime: Time,
    gaia_sources: list[GaiaSource],
) -> VerdictResult:
    """Compare the difference-image centroid against the target's own catalog position."""
    matched = match_target_gaia_source(catalog_ra, catalog_dec, gaia_sources)
    if matched is not None:
        target_coord = propagate_position(matched, obstime)
        pm_note = f"target matched to Gaia DR3 {matched.source_id}, proper motion propagated to obs epoch"
    else:
        target_coord = SkyCoord(ra=catalog_ra * u.deg, dec=catalog_dec * u.deg)
        pm_note = (
            f"no Gaia DR3 counterpart found within {TARGET_GAIA_MATCH_RADIUS_ARCSEC:g} arcsec "
            "of catalog position; using catalog position as-is (no proper-motion correction)"
        )

    centroid_coord = SkyCoord(ra=centroid_ra * u.deg, dec=centroid_dec * u.deg)
    offset_arcsec = float(target_coord.separation(centroid_coord).arcsec)

    scale = pixel_scale_arcsec(wcs)
    if not np.isfinite(centroid_uncertainty_col) or not np.isfinite(centroid_uncertainty_row):
        return VerdictResult(
            target_position={"ra": target_coord.ra.deg, "dec": target_coord.dec.deg},
            centroid_offset_arcsec=offset_arcsec,
            centroid_offset_sigma=float("nan"),
            verdict=VERDICT_INCONCLUSIVE,
            verdict_reason=(
                "centroid positional uncertainty could not be estimated from "
                "cadence-resampling bootstrap (too few stable resamples); refusing to "
                "force an on/off-target call"
            ),
        )

    uncertainty_arcsec = scale * float(np.hypot(centroid_uncertainty_col, centroid_uncertainty_row))
    if uncertainty_arcsec <= 0:
        return VerdictResult(
            target_position={"ra": target_coord.ra.deg, "dec": target_coord.dec.deg},
            centroid_offset_arcsec=offset_arcsec,
            centroid_offset_sigma=float("nan"),
            verdict=VERDICT_INCONCLUSIVE,
            verdict_reason="bootstrap centroid uncertainty collapsed to zero; refusing to force a call",
        )

    sigma = offset_arcsec / uncertainty_arcsec
    if sigma >= CENTROID_OFFSET_SIGMA_THRESHOLD:
        verdict = VERDICT_OFF_TARGET_BLEND
        reason = (
            f"difference-image centroid is offset {offset_arcsec:.2f}\" "
            f"({sigma:.1f}sigma) from the target's own position -- the transit "
            f"signal is not coming from the target. {pm_note}."
        )
    else:
        verdict = VERDICT_ON_TARGET
        reason = (
            f"difference-image centroid is offset {offset_arcsec:.2f}\" "
            f"({sigma:.1f}sigma) from the target's own position, below the "
            f"{CENTROID_OFFSET_SIGMA_THRESHOLD}sigma threshold -- consistent with "
            f"the transit coming from the target. {pm_note}."
        )

    return VerdictResult(
        target_position={"ra": target_coord.ra.deg, "dec": target_coord.dec.deg},
        centroid_offset_arcsec=offset_arcsec,
        centroid_offset_sigma=sigma,
        verdict=verdict,
        verdict_reason=reason,
    )
