import pytest
from astropy.table import Table

from localizr import gaia as gaia_module


class _FakeJob:
    def __init__(self, table):
        self._table = table

    def get_results(self):
        return self._table


def _fake_table():
    return Table(
        {
            "source_id": [2, 1, 3],
            "ra": [10.001, 10.0001, 10.01],
            "dec": [20.0, 20.0, 20.0],
            "phot_g_mean_mag": [15.0, 12.0, 18.0],
            "pmra": [1.0, 2.0, 3.0],
            "pmdec": [0.5, 0.6, 0.7],
            "parallax": [1.0, 2.0, 3.0],
            "ref_epoch": [2016.0, 2016.0, 2016.0],
            "dist": [0.002, 0.0004, 0.01],  # degrees; smallest belongs to source_id=1
        }
    )


def test_query_nearby_sources_sorts_by_separation(monkeypatch):
    monkeypatch.setattr(
        gaia_module.Gaia, "cone_search_async", lambda coord, radius: _FakeJob(_fake_table())
    )

    sources = gaia_module.query_nearby_sources(10.0, 20.0, 30.0)

    assert [s.source_id for s in sources] == [1, 2, 3]
    assert sources[0].sep_arcsec == pytest.approx(0.0004 * 3600)
    assert sources[0].gmag == pytest.approx(12.0)
    assert sources[0].pmra == pytest.approx(2.0)


def test_gaia_source_to_dict_shape():
    source = gaia_module.GaiaSource(
        source_id=1, ra=10.0, dec=20.0, sep_arcsec=1.5, gmag=12.0,
        pmra=1.0, pmdec=1.0, parallax=1.0, ref_epoch=2016.0,
    )
    d = source.to_dict()
    assert set(d.keys()) == {"source_id", "ra", "dec", "sep_arcsec", "gmag"}


def test_query_nearby_sources_wraps_errors(monkeypatch):
    def raiser(coord, radius):
        raise RuntimeError("boom")

    monkeypatch.setattr(gaia_module.Gaia, "cone_search_async", raiser)

    with pytest.raises(gaia_module.GaiaQueryError):
        gaia_module.query_nearby_sources(10.0, 20.0, 30.0)
