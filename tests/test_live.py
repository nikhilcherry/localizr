"""Real MAST + Gaia network tests. Skipped by default; run with `pytest -m network`.

These are the project's actual verification fixtures:

- KIC 4281068 / KOI K07689.01 is catalog-selected specifically because Kepler
  DR25's Robovetter flags it `koi_fpflag_co` (centroid offset) -- it's a
  known blend. localizr must call it `off_target_blend`.
- TIC 218795833 (TOI-519.01, "TOI-519 b") is a `CP`-dispositioned confirmed
  hot Jupiter transiting an isolated M dwarf, with a deep (~11%) transit for
  high per-pixel signal-to-noise. localizr must call it `on_target`.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import localizr

pytestmark = pytest.mark.network

KIC_BLEND = dict(kic_id=4281068, period=1.01530474, epoch_btjd=131.517946, duration_hours=8.7622)
TIC_CLEAN = dict(tic_id=218795833, period=1.2652338, epoch_btjd=1491.8770560, duration_hours=1.2121003)


def test_blend_fixture_is_off_target_blend():
    result = localizr.localize(**KIC_BLEND)
    assert result.verdict == "off_target_blend", result.verdict_reason
    assert result.centroid_offset_sigma >= localizr.CENTROID_OFFSET_SIGMA_THRESHOLD


def test_clean_confirmed_planet_is_on_target():
    result = localizr.localize(**TIC_CLEAN)
    assert result.verdict == "on_target", result.verdict_reason
    assert result.centroid_offset_sigma < localizr.CENTROID_OFFSET_SIGMA_THRESHOLD


def test_gaia_cross_match_returns_sane_sources():
    result = localizr.localize(**KIC_BLEND)
    assert len(result.nearby_gaia_sources) >= 1
    seps = [s["sep_arcsec"] for s in result.nearby_gaia_sources]
    assert seps == sorted(seps)
    assert all(0 <= s < 200 for s in seps)


def test_cwd_independence_via_subprocess(tmp_path):
    other_cwd = tmp_path / "somewhere_else"
    other_cwd.mkdir()
    json_path = tmp_path / "result.json"

    proc = subprocess.run(
        [
            sys.executable, "-m", "localizr.cli", "localize",
            "--kic-id", str(KIC_BLEND["kic_id"]),
            "--period", str(KIC_BLEND["period"]),
            "--epoch", str(KIC_BLEND["epoch_btjd"]),
            "--duration-hours", str(KIC_BLEND["duration_hours"]),
            "--json", "--json-path", str(json_path),
        ],
        cwd=str(other_cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode in (0, 1), proc.stderr
    data = json.loads(json_path.read_text())
    assert data["verdict"] == "off_target_blend"


def test_inconclusive_on_duration_far_longer_than_period():
    bad = dict(KIC_BLEND)
    bad["duration_hours"] = KIC_BLEND["period"] * 24 * 5  # nonsensically long
    result = localizr.localize(**bad)
    assert result.verdict == "inconclusive"
