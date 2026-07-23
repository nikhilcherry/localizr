from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

from localizr.plots import save_plot


def _simple_wcs(scale_deg: float = 0.001104) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [100.0, 20.0]
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cdelt = [-scale_deg, scale_deg]
    return w


def test_save_plot_creates_missing_parent_directories(tmp_path):
    # A user passing --plot-path into a directory that doesn't exist yet
    # (a common case: a fresh output dir for a batch of targets) should get
    # the plot saved there, not a raw FileNotFoundError from matplotlib.
    out_path = tmp_path / "nested" / "does" / "not" / "exist" / "plot.png"
    diff_image = np.random.default_rng(0).random((5, 5))

    result_path = save_plot(
        str(out_path), diff_image, _simple_wcs(),
        target_ra=100.0, target_dec=20.0,
        centroid_ra=100.0001, centroid_dec=20.0001,
        gaia_sources=[],
    )

    assert result_path == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
