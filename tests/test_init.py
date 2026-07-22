import json
from pathlib import Path

import localizr
from localizr import gaia as gaia_module

FIXTURE = Path(__file__).parent / "fixtures" / "kic4281068_q1_tpf.fits.gz"

PERIOD = 1.01530474
EPOCH_BKJD = 131.517946
DURATION_HOURS = 8.7622


def _no_gaia(monkeypatch):
    monkeypatch.setattr(gaia_module, "query_nearby_sources", lambda ra, dec, radius: [])


def test_localize_offline_with_tpf_path(monkeypatch):
    _no_gaia(monkeypatch)
    result = localizr.localize(
        tpf_path=str(FIXTURE), period=PERIOD, epoch_btjd=EPOCH_BKJD, duration_hours=DURATION_HOURS,
    )
    assert result.mission == "Kepler"
    assert result.kic_id == 4281068
    assert result.tic_id is None
    assert result.verdict in {"on_target", "off_target_blend", "inconclusive"}
    assert result.difference_image_centroid["ra"] is not None
    assert result.difference_image_centroid["dec"] is not None


def test_localize_inconclusive_when_duration_not_shorter_than_period(monkeypatch):
    _no_gaia(monkeypatch)
    result = localizr.localize(
        tpf_path=str(FIXTURE), period=PERIOD, epoch_btjd=EPOCH_BKJD,
        duration_hours=PERIOD * 24 * 2,  # duration >= period: physically nonsensical
    )
    assert result.verdict == "inconclusive"
    assert result.centroid_offset_arcsec is None
    assert result.centroid_offset_sigma is None
    assert result.difference_image_centroid == {"ra": None, "dec": None}


def test_localization_result_to_json_roundtrip(monkeypatch, tmp_path):
    _no_gaia(monkeypatch)
    result = localizr.localize(
        tpf_path=str(FIXTURE), period=PERIOD, epoch_btjd=EPOCH_BKJD, duration_hours=DURATION_HOURS,
    )
    out_path = tmp_path / "out.json"
    text = result.to_json(str(out_path))
    parsed = json.loads(text)
    assert parsed["mission"] == "Kepler"
    assert parsed["kic_id"] == 4281068
    assert out_path.read_text() == text


def test_localization_result_save_plot(monkeypatch, tmp_path):
    _no_gaia(monkeypatch)
    result = localizr.localize(
        tpf_path=str(FIXTURE), period=PERIOD, epoch_btjd=EPOCH_BKJD, duration_hours=DURATION_HOURS,
    )
    out_path = tmp_path / "diff.png"
    result.save_plot(str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_localization_result_save_plot_raises_when_inconclusive_with_no_image(monkeypatch):
    _no_gaia(monkeypatch)
    result = localizr.localize(
        tpf_path=str(FIXTURE), period=PERIOD, epoch_btjd=EPOCH_BKJD,
        duration_hours=PERIOD * 24 * 2,
    )
    try:
        result.save_plot("/tmp/should_not_be_created.png")
        raised = False
    except localizr.LocalizationError:
        raised = True
    assert raised
