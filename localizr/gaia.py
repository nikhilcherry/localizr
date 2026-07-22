"""Gaia DR3 cross-match within a target pixel file's footprint."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

GAIA_TABLE = "gaiadr3.gaia_source"


class GaiaQueryError(Exception):
    """Raised when the Gaia archive can't be queried."""


@dataclass
class GaiaSource:
    source_id: int
    ra: float
    dec: float
    sep_arcsec: float
    gmag: float | None
    pmra: float | None
    pmdec: float | None
    parallax: float | None
    ref_epoch: float

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "ra": self.ra,
            "dec": self.dec,
            "sep_arcsec": self.sep_arcsec,
            "gmag": self.gmag,
        }


def query_nearby_sources(ra_deg: float, dec_deg: float, radius_arcsec: float) -> list[GaiaSource]:
    """Query Gaia DR3 for all sources within ``radius_arcsec`` of (ra, dec).

    Returns sources sorted by separation from (ra, dec), nearest first.
    """
    Gaia.MAIN_GAIA_TABLE = GAIA_TABLE
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame="icrs")
    try:
        job = Gaia.cone_search_async(coord, radius=u.Quantity(radius_arcsec, u.arcsec))
        table = job.get_results()
    except Exception as exc:  # noqa: BLE001 - surface any TAP/network failure uniformly
        raise GaiaQueryError(f"Gaia DR3 cone search failed: {exc}") from exc

    sources = []
    for row in table:
        sources.append(
            GaiaSource(
                source_id=int(row["source_id"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                sep_arcsec=float(row["dist"]) * 3600.0,
                gmag=float(row["phot_g_mean_mag"]) if row["phot_g_mean_mag"] is not None
                and not _is_masked(row["phot_g_mean_mag"]) else None,
                pmra=_maybe_float(row["pmra"]),
                pmdec=_maybe_float(row["pmdec"]),
                parallax=_maybe_float(row["parallax"]),
                ref_epoch=float(row["ref_epoch"]),
            )
        )
    sources.sort(key=lambda s: s.sep_arcsec)
    return sources


def _is_masked(value) -> bool:
    return hasattr(value, "mask") and bool(value.mask)


def _maybe_float(value) -> float | None:
    if value is None or _is_masked(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
