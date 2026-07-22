"""Diagnostic figure: difference image with target, centroid, and Gaia overlays."""

from __future__ import annotations

import matplotlib.pyplot as plt


def make_diagnostic_figure(
    diff_image,
    wcs,
    target_ra: float,
    target_dec: float,
    centroid_ra: float,
    centroid_dec: float,
    gaia_sources: list[dict],
    title: str | None = None,
):
    target_col, target_row = wcs.all_world2pix(target_ra, target_dec, 0)
    centroid_col, centroid_row = wcs.all_world2pix(centroid_ra, centroid_dec, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(diff_image, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="out-of-transit minus in-transit flux")

    for source in gaia_sources:
        col, row = wcs.all_world2pix(source["ra"], source["dec"], 0)
        ax.plot(col, row, marker="o", mfc="none", mec="orange", mew=1.5, ms=11)
        ax.annotate(
            f"{source['sep_arcsec']:.1f}\"",
            (col, row),
            color="orange",
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.plot(
        target_col, target_row, marker="*", color="cyan", mec="black", ms=18,
        linestyle="none", label="target (catalog position)",
    )
    ax.plot(
        centroid_col, centroid_row, marker="x", color="red", mew=3, ms=13,
        linestyle="none", label="difference-image centroid",
    )

    ax.set_xlabel("column (pixel)")
    ax.set_ylabel("row (pixel)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def save_plot(
    path,
    diff_image,
    wcs,
    target_ra: float,
    target_dec: float,
    centroid_ra: float,
    centroid_dec: float,
    gaia_sources: list[dict],
    title: str | None = None,
) -> str:
    fig = make_diagnostic_figure(
        diff_image, wcs, target_ra, target_dec, centroid_ra, centroid_dec, gaia_sources, title
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
