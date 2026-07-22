from pathlib import Path

import pytest

from localizr.tpf import TpfResolutionError, resolve_tpf

FIXTURE = Path(__file__).parent / "fixtures" / "kic4281068_q1_tpf.fits.gz"


def test_resolve_tpf_rejects_no_target_given():
    with pytest.raises(TpfResolutionError):
        resolve_tpf()


def test_resolve_tpf_rejects_multiple_targets_given():
    with pytest.raises(TpfResolutionError):
        resolve_tpf(tic_id=1, kic_id=2)
    with pytest.raises(TpfResolutionError):
        resolve_tpf(tic_id=1, tpf_path="foo.fits")


def test_resolve_tpf_reads_local_kepler_fixture():
    tpf, mission, catalog_id = resolve_tpf(tpf_path=str(FIXTURE))
    assert mission == "Kepler"
    assert catalog_id == 4281068
    assert tpf.flux.shape == (1626, 5, 6)


def test_resolve_tpf_bad_local_path_raises():
    with pytest.raises(TpfResolutionError):
        resolve_tpf(tpf_path="/nonexistent/path/does_not_exist.fits")
