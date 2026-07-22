import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

from localizr.gaia import GaiaSource
from localizr.verdict import (
    TARGET_GAIA_MATCH_RADIUS_ARCSEC,
    VERDICT_INCONCLUSIVE,
    VERDICT_OFF_TARGET_BLEND,
    VERDICT_ON_TARGET,
    compute_verdict,
    match_target_gaia_source,
    pixel_scale_arcsec,
    propagate_position,
)


def _simple_wcs(scale_deg: float = 0.001104) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [100.0, 20.0]
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cdelt = [-scale_deg, scale_deg]
    return w


def _gaia_source(source_id, ra, dec, pmra=0.0, pmdec=0.0, parallax=10.0, gmag=12.0):
    return GaiaSource(
        source_id=source_id, ra=ra, dec=dec, sep_arcsec=0.0,
        gmag=gmag, pmra=pmra, pmdec=pmdec, parallax=parallax, ref_epoch=2016.0,
    )


def test_pixel_scale_arcsec():
    scale = pixel_scale_arcsec(_simple_wcs(0.001104))
    assert scale == pytest.approx(0.001104 * 3600, rel=1e-3)


def test_match_target_gaia_source_within_radius():
    sources = [_gaia_source(1, 100.0005, 20.0005), _gaia_source(2, 100.05, 20.05)]
    matched = match_target_gaia_source(100.0, 20.0, sources, max_sep_arcsec=5.0)
    assert matched is not None
    assert matched.source_id == 1


def test_match_target_gaia_source_none_within_radius():
    sources = [_gaia_source(1, 100.05, 20.05)]
    matched = match_target_gaia_source(100.0, 20.0, sources, max_sep_arcsec=1.0)
    assert matched is None


def test_match_target_gaia_source_empty_list():
    assert match_target_gaia_source(100.0, 20.0, [], max_sep_arcsec=5.0) is None


def test_propagate_position_zero_pm_is_static():
    source = _gaia_source(1, 100.0, 20.0, pmra=0.0, pmdec=0.0)
    coord = propagate_position(source, Time(2025.0, format="jyear"))
    assert coord.ra.deg == pytest.approx(100.0, abs=1e-9)
    assert coord.dec.deg == pytest.approx(20.0, abs=1e-9)


def test_propagate_position_nonzero_pm_moves_star():
    source = _gaia_source(1, 100.0, 20.0, pmra=500.0, pmdec=0.0)  # 500 mas/yr
    coord = propagate_position(source, Time(2026.0, format="jyear"))  # 10 yr later
    orig = SkyCoord(ra=100.0 * u.deg, dec=20.0 * u.deg)
    sep_arcsec = orig.separation(coord).arcsec
    assert 4.5 < sep_arcsec < 5.5  # 500 mas/yr * 10 yr = 5000 mas = 5"


def test_compute_verdict_no_gaia_match_note_reflects_match_radius():
    result = compute_verdict(
        catalog_ra=100.0, catalog_dec=20.0,
        centroid_ra=100.0, centroid_dec=20.00002,
        centroid_uncertainty_col=0.3, centroid_uncertainty_row=0.3,
        wcs=_simple_wcs(), obstime=Time(2020.0, format="jyear"), gaia_sources=[],
    )
    assert f"{TARGET_GAIA_MATCH_RADIUS_ARCSEC:g} arcsec" in result.verdict_reason


def test_compute_verdict_on_target_small_offset():
    result = compute_verdict(
        catalog_ra=100.0, catalog_dec=20.0,
        centroid_ra=100.0, centroid_dec=20.00002,  # ~0.07"
        centroid_uncertainty_col=0.3, centroid_uncertainty_row=0.3,
        wcs=_simple_wcs(), obstime=Time(2020.0, format="jyear"), gaia_sources=[],
    )
    assert result.verdict == VERDICT_ON_TARGET
    assert result.centroid_offset_sigma < 3.0


def test_compute_verdict_off_target_large_offset():
    result = compute_verdict(
        catalog_ra=100.0, catalog_dec=20.0,
        centroid_ra=100.01, centroid_dec=20.0,  # ~34"
        centroid_uncertainty_col=0.2, centroid_uncertainty_row=0.2,
        wcs=_simple_wcs(), obstime=Time(2020.0, format="jyear"), gaia_sources=[],
    )
    assert result.verdict == VERDICT_OFF_TARGET_BLEND
    assert result.centroid_offset_sigma >= 3.0


def test_compute_verdict_inconclusive_on_nan_uncertainty():
    result = compute_verdict(
        catalog_ra=100.0, catalog_dec=20.0,
        centroid_ra=100.01, centroid_dec=20.0,
        centroid_uncertainty_col=float("nan"), centroid_uncertainty_row=float("nan"),
        wcs=_simple_wcs(), obstime=Time(2020.0, format="jyear"), gaia_sources=[],
    )
    assert result.verdict == VERDICT_INCONCLUSIVE
    assert np.isnan(result.centroid_offset_sigma)


def test_compute_verdict_matches_target_to_gaia_and_propagates_pm():
    matching_source = _gaia_source(999, 100.0, 20.0, pmra=0.0, pmdec=0.0)
    result = compute_verdict(
        catalog_ra=100.0, catalog_dec=20.0,
        centroid_ra=100.0, centroid_dec=20.0,
        centroid_uncertainty_col=0.3, centroid_uncertainty_row=0.3,
        wcs=_simple_wcs(), obstime=Time(2020.0, format="jyear"),
        gaia_sources=[matching_source],
    )
    assert "999" in result.verdict_reason
