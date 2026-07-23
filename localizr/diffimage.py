"""In-transit-minus-out-of-transit difference imaging and flux-weighted centroiding.

This is the core numerical technique behind Kepler DR25's centroid-offset
vetting (``koi_fpflag_co``): average the pixels while the target is in
transit, average them while it isn't, subtract, and find the flux-weighted
centroid of what's left. If the transit signal truly comes from the target,
that centroid lands on the target's own pixel. If it comes from a blended
neighbor, the centroid is pulled toward the neighbor instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning

MIN_IN_TRANSIT_CADENCES = 3
MIN_OUT_OF_TRANSIT_CADENCES = 3
MIN_VALID_PIXELS = 4
# Fractional buffer (relative to the transit duration) left as a gap on
# either side of the transit window before a cadence counts as
# "out-of-transit", so ingress/egress smearing doesn't contaminate either mean.
OUT_OF_TRANSIT_BUFFER_FACTOR = 1.0
BOOTSTRAP_RESAMPLES = 200
# Per-pixel sigma-clipping threshold applied across cadences before
# averaging, so a single contaminated cadence (cosmic ray hit, unflagged
# flare, momentum-dump-adjacent artifact the mission's own quality bitmask
# missed) can't pull the in-transit/out-of-transit mean and, downstream, the
# centroid. Below this many cadences, clipping statistics aren't meaningful,
# so the plain mean is used instead.
CADENCE_SIGMA_CLIP = 5.0
MIN_CADENCES_FOR_SIGMA_CLIP = 5


class DifferenceImageError(Exception):
    """Raised when a difference image or centroid cannot be computed."""


@dataclass
class DifferenceImageResult:
    diff_image: np.ndarray
    in_transit_image: np.ndarray
    out_of_transit_image: np.ndarray
    in_transit_mask: np.ndarray
    out_of_transit_mask: np.ndarray
    centroid_col: float
    centroid_row: float
    centroid_uncertainty_col: float
    centroid_uncertainty_row: float

    @property
    def n_in_transit(self) -> int:
        return int(np.count_nonzero(self.in_transit_mask))

    @property
    def n_out_of_transit(self) -> int:
        return int(np.count_nonzero(self.out_of_transit_mask))


def transit_masks(
    time: np.ndarray, period: float, epoch: float, duration_hours: float
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (in_transit, out_of_transit) masks for cadences in ``time``.

    ``time``, ``period``, and ``epoch`` must all be in the same time system
    (e.g. all BTJD, or all BKJD) -- whatever units the TPF's own time column
    uses.
    """
    if duration_hours <= 0:
        raise DifferenceImageError(f"duration_hours must be positive, got {duration_hours}")
    if period <= 0:
        raise DifferenceImageError(f"period must be positive, got {period}")

    half_duration_days = duration_hours / 24.0 / 2.0
    if 2 * half_duration_days >= period:
        raise DifferenceImageError(
            f"duration_hours ({duration_hours}h) is not shorter than period "
            f"({period}d) -- in-transit and out-of-transit cadences can't be "
            "separated"
        )

    # Signed offset from the nearest transit center, folded into [-P/2, P/2).
    delta = np.mod(time - epoch + period / 2.0, period) - period / 2.0

    in_transit = np.abs(delta) <= half_duration_days
    out_of_transit = np.abs(delta) > half_duration_days * (1.0 + OUT_OF_TRANSIT_BUFFER_FACTOR)
    return in_transit, out_of_transit


def _nanmean_over_cadences(flux: np.ndarray) -> np.ndarray:
    """np.nanmean(flux, axis=0), quietly -- some border pixels in a real TPF
    cutout are NaN for every cadence (never collected by the pipeline), which
    is expected and handled downstream, not a bug worth warning about."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(flux, axis=0)


def _robust_mean_over_cadences(flux: np.ndarray, sigma: float = CADENCE_SIGMA_CLIP) -> np.ndarray:
    """Per-pixel iterative sigma-clipped mean across the cadence axis.

    Each pixel is clipped independently (an artifact in one corner of the
    cutout shouldn't cost the rest of the array any cadences), matching how
    ``astropy.stats.sigma_clip`` treats an ``axis`` argument. Cadence quality
    bitmasks already filter out most known-bad cadences before this ever
    runs; this catches the outliers that slip through unflagged.
    """
    if flux.shape[0] < MIN_CADENCES_FOR_SIGMA_CLIP:
        return _nanmean_over_cadences(flux)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        clipped = sigma_clip(flux, sigma=sigma, axis=0, masked=True)
        mean = np.ma.mean(clipped, axis=0)
        return np.ma.filled(mean, np.nan)


def flux_weighted_centroid(image: np.ndarray) -> tuple[float, float]:
    """Flux-weighted (col, row) centroid of a 2-D image.

    Only positive flux contributes -- negative difference-image pixels
    (regions that got *brighter* in transit, e.g. noise or a bright neighbor
    unaffected by this transit) are clipped to zero rather than allowed to
    pull the centroid around.
    """
    clipped = np.where(np.isfinite(image), np.clip(image, 0.0, None), 0.0)
    total = clipped.sum()
    if not np.isfinite(total) or total <= 0:
        raise DifferenceImageError("difference image has no positive flux to centroid on")

    ny, nx = image.shape
    rows, cols = np.indices((ny, nx))
    row_c = float((rows * clipped).sum() / total)
    col_c = float((cols * clipped).sum() / total)
    return col_c, row_c


def compute_difference_image(
    flux: np.ndarray,
    time: np.ndarray,
    period: float,
    epoch: float,
    duration_hours: float,
    quality: np.ndarray | None = None,
) -> DifferenceImageResult:
    """Build the difference image and its flux-weighted centroid.

    ``flux`` is a (n_cadence, ny, nx) array (units don't matter -- only
    relative flux is used). ``quality`` is an optional per-cadence bitmask;
    nonzero cadences are excluded, matching the pipeline's own quality flags.
    The in-transit and out-of-transit means are each computed with per-pixel
    sigma-clipping across cadences (see :func:`_robust_mean_over_cadences`),
    so a single unflagged bad cadence can't dominate either mean.
    """
    flux = np.asarray(flux, dtype=float)
    time = np.asarray(time, dtype=float)
    if flux.ndim != 3:
        raise DifferenceImageError(f"flux must be 3-D (cadence, row, col), got shape {flux.shape}")
    if flux.shape[0] != time.shape[0]:
        raise DifferenceImageError(
            f"flux has {flux.shape[0]} cadences but time has {time.shape[0]}"
        )

    good = np.isfinite(time)
    if quality is not None:
        good &= np.asarray(quality) == 0

    in_transit, out_of_transit = transit_masks(time, period, epoch, duration_hours)
    in_transit &= good
    out_of_transit &= good

    n_valid_pixels = int(np.count_nonzero(np.any(np.isfinite(flux) & (flux != 0), axis=0)))
    if n_valid_pixels < MIN_VALID_PIXELS:
        raise DifferenceImageError(
            f"only {n_valid_pixels} usable pixels in TPF, need at least {MIN_VALID_PIXELS}"
        )

    n_in = int(np.count_nonzero(in_transit))
    n_out = int(np.count_nonzero(out_of_transit))
    if n_in < MIN_IN_TRANSIT_CADENCES:
        raise DifferenceImageError(
            f"only {n_in} in-transit cadences available, need at least {MIN_IN_TRANSIT_CADENCES}"
        )
    if n_out < MIN_OUT_OF_TRANSIT_CADENCES:
        raise DifferenceImageError(
            f"only {n_out} out-of-transit cadences available, need at least "
            f"{MIN_OUT_OF_TRANSIT_CADENCES}"
        )

    in_image = _robust_mean_over_cadences(flux[in_transit])
    out_image = _robust_mean_over_cadences(flux[out_of_transit])
    diff_image = out_image - in_image

    col_c, row_c = flux_weighted_centroid(diff_image)
    unc_col, unc_row = _bootstrap_centroid_uncertainty(flux, in_transit, out_of_transit)

    return DifferenceImageResult(
        diff_image=diff_image,
        in_transit_image=in_image,
        out_of_transit_image=out_image,
        in_transit_mask=in_transit,
        out_of_transit_mask=out_of_transit,
        centroid_col=col_c,
        centroid_row=row_c,
        centroid_uncertainty_col=unc_col,
        centroid_uncertainty_row=unc_row,
    )


def _bootstrap_centroid_uncertainty(
    flux: np.ndarray,
    in_transit: np.ndarray,
    out_of_transit: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Empirical centroid uncertainty via cadence-resampling bootstrap.

    Repeatedly resamples (with replacement) the in-transit and
    out-of-transit cadence sets, rebuilds the difference image, and
    recomputes the centroid. The spread across resamples stands in for a
    formal photon-noise covariance, which would require PSF modeling this
    tool deliberately doesn't do (see README non-goals).
    """
    rng = rng or np.random.default_rng(0)
    in_idx = np.flatnonzero(in_transit)
    out_idx = np.flatnonzero(out_of_transit)

    cols, rows = [], []
    for _ in range(n_resamples):
        in_sample = rng.choice(in_idx, size=len(in_idx), replace=True)
        out_sample = rng.choice(out_idx, size=len(out_idx), replace=True)
        diff = _nanmean_over_cadences(flux[out_sample]) - _nanmean_over_cadences(flux[in_sample])
        try:
            c, r = flux_weighted_centroid(diff)
        except DifferenceImageError:
            continue
        cols.append(c)
        rows.append(r)

    if len(cols) < n_resamples // 4:
        # Too many degenerate resamples to trust a spread; signal this with NaN
        # so callers can treat it as "uncertainty could not be estimated".
        return float("nan"), float("nan")
    return float(np.std(cols)), float(np.std(rows))
