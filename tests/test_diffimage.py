import numpy as np
import pytest

from localizr.diffimage import (
    DifferenceImageError,
    compute_difference_image,
    flux_weighted_centroid,
    transit_masks,
)


def test_transit_masks_basic():
    time = np.linspace(0, 10, 1001)
    period, epoch, duration_hours = 2.0, 1.0, 2.4

    in_mask, out_mask = transit_masks(time, period, epoch, duration_hours)

    assert in_mask.sum() > 0
    assert out_mask.sum() > 0
    assert not np.any(in_mask & out_mask)

    half_duration_days = duration_hours / 24.0 / 2.0
    phase = np.mod(time[in_mask] - epoch + period / 2.0, period) - period / 2.0
    assert np.all(np.abs(phase) <= half_duration_days + 1e-9)


def test_transit_masks_rejects_duration_not_shorter_than_period():
    time = np.linspace(0, 10, 100)
    with pytest.raises(DifferenceImageError):
        transit_masks(time, period=1.0, epoch=0.0, duration_hours=30.0)


def test_transit_masks_rejects_nonpositive_inputs():
    time = np.linspace(0, 10, 100)
    with pytest.raises(DifferenceImageError):
        transit_masks(time, period=-1.0, epoch=0.0, duration_hours=1.0)
    with pytest.raises(DifferenceImageError):
        transit_masks(time, period=1.0, epoch=0.0, duration_hours=0.0)


def test_flux_weighted_centroid_recovers_known_peak():
    image = np.zeros((10, 10))
    image[3, 7] = 100.0  # row=3, col=7
    col, row = flux_weighted_centroid(image)
    assert np.isclose(col, 7.0)
    assert np.isclose(row, 3.0)


def test_flux_weighted_centroid_ignores_negative_flux():
    image = np.full((5, 5), -10.0)
    image[2, 2] = 50.0
    col, row = flux_weighted_centroid(image)
    assert np.isclose(col, 2.0)
    assert np.isclose(row, 2.0)


def test_flux_weighted_centroid_rejects_all_nonpositive():
    image = -np.ones((5, 5))
    with pytest.raises(DifferenceImageError):
        flux_weighted_centroid(image)


def _synthetic_tpf_cube(
    n_cadence, ny, nx, source_row, source_col, period, epoch, duration_hours,
    depth=0.2, background=1000.0, source_flux=500.0, seed=42,
):
    """A TPF-shaped flux cube with a single dimming source at a known pixel."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 20, n_cadence)
    in_mask, _ = transit_masks(time, period, epoch, duration_hours)

    flux = np.full((n_cadence, ny, nx), background, dtype=float)
    flux[:, source_row, source_col] += source_flux
    flux[in_mask, source_row, source_col] -= source_flux * depth
    flux += rng.normal(0, 1.0, size=flux.shape)
    return flux, time


def test_compute_difference_image_recovers_injected_offset_on_target():
    ny, nx = 9, 9
    source_row, source_col = 4, 4  # center pixel -- simulates an on-target signal
    period, epoch, duration_hours = 3.0, 1.0, 4.0

    flux, time = _synthetic_tpf_cube(400, ny, nx, source_row, source_col, period, epoch, duration_hours)
    result = compute_difference_image(flux, time, period, epoch, duration_hours)

    assert result.centroid_col == pytest.approx(source_col, abs=0.3)
    assert result.centroid_row == pytest.approx(source_row, abs=0.3)
    assert result.n_in_transit >= 3
    assert result.n_out_of_transit >= 3


def test_compute_difference_image_recovers_injected_offset_blend():
    ny, nx = 9, 9
    source_row, source_col = 7, 1  # off-center -- simulates a blended neighbor
    period, epoch, duration_hours = 3.0, 1.0, 4.0

    flux, time = _synthetic_tpf_cube(400, ny, nx, source_row, source_col, period, epoch, duration_hours)
    result = compute_difference_image(flux, time, period, epoch, duration_hours)

    assert result.centroid_col == pytest.approx(source_col, abs=0.3)
    assert result.centroid_row == pytest.approx(source_row, abs=0.3)
    # and it should clearly NOT land on the array center (where a target usually sits)
    assert abs(result.centroid_col - nx / 2) > 1.0


def test_compute_difference_image_is_robust_to_a_single_bad_cadence():
    """A cosmic-ray-like spike on one out-of-transit cadence, far from the real
    source, must not drag the centroid off target -- this is what the
    per-pixel sigma-clipped cadence mean is for."""
    ny, nx = 9, 9
    source_row, source_col = 4, 4
    period, epoch, duration_hours = 3.0, 1.0, 4.0

    flux, time = _synthetic_tpf_cube(400, ny, nx, source_row, source_col, period, epoch, duration_hours)
    _, out_mask = transit_masks(time, period, epoch, duration_hours)
    bad_cadence = np.flatnonzero(out_mask)[10]
    flux[bad_cadence, 0, nx - 1] += 50_000.0  # spike in a far corner pixel

    result = compute_difference_image(flux, time, period, epoch, duration_hours)

    assert result.centroid_col == pytest.approx(source_col, abs=0.3)
    assert result.centroid_row == pytest.approx(source_row, abs=0.3)


def test_compute_difference_image_too_few_cadences_is_an_error():
    ny, nx = 5, 5
    flux = np.full((5, ny, nx), 100.0)
    time = np.array([0.0, 0.01, 5.0, 5.01, 10.0])
    with pytest.raises(DifferenceImageError):
        compute_difference_image(flux, time, period=1.0, epoch=0.0, duration_hours=1.0)


def test_compute_difference_image_too_few_pixels_is_an_error():
    flux = np.full((100, 1, 1), 100.0)
    time = np.linspace(0, 20, 100)
    with pytest.raises(DifferenceImageError):
        compute_difference_image(flux, time, period=3.0, epoch=1.0, duration_hours=4.0)


def test_compute_difference_image_rejects_mismatched_lengths():
    flux = np.zeros((10, 5, 5))
    time = np.zeros(5)
    with pytest.raises(DifferenceImageError):
        compute_difference_image(flux, time, period=1.0, epoch=0.0, duration_hours=1.0)
