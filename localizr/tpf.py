"""Target pixel file resolution: fetch by TIC/KIC id (with local caching) or load from disk."""

from __future__ import annotations

import os
from pathlib import Path

import lightkurve as lk

DEFAULT_CACHE_DIR = Path.home() / ".localizr" / "tpf_cache"


class TpfResolutionError(Exception):
    """Raised when a target pixel file cannot be located, searched for, or read."""


def _cache_dir(cache_dir: str | os.PathLike | None) -> Path:
    path = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prefer_author(search_result, preferred: str):
    """Narrow a lightkurve SearchResult to a preferred pipeline author, if present."""
    mask = search_result.author == preferred
    if mask.any():
        return search_result[mask]
    return search_result


def _first_by_sequence(search_result):
    """Return the search result sorted so the earliest sector/quarter is first."""
    table = search_result.table
    seq_col = "sequence_number" if "sequence_number" in table.colnames else None
    if seq_col is None:
        return search_result
    order = table[seq_col].argsort()
    return search_result[order]


def resolve_tpf(
    *,
    tic_id: int | str | None = None,
    kic_id: int | str | None = None,
    tpf_path: str | os.PathLike | None = None,
    sector: int | None = None,
    quarter: int | None = None,
    cache_dir: str | os.PathLike | None = None,
):
    """Resolve a target pixel file from a TIC id, a KIC id, or a local path.

    Exactly one of ``tic_id``, ``kic_id``, ``tpf_path`` must be given. Returns
    ``(tpf, mission, catalog_id)`` where ``mission`` is ``"TESS"`` or
    ``"Kepler"`` and ``catalog_id`` is the resolved TIC/KIC id (as an int).
    """
    given = [name for name, val in (("tic_id", tic_id), ("kic_id", kic_id), ("tpf_path", tpf_path)) if val is not None]
    if len(given) != 1:
        raise TpfResolutionError(
            f"exactly one of tic_id, kic_id, tpf_path must be given, got: {given or 'none'}"
        )

    if tpf_path is not None:
        try:
            tpf = lk.read(str(tpf_path))
        except Exception as exc:  # noqa: BLE001 - surface any read failure uniformly
            raise TpfResolutionError(f"could not read TPF at '{tpf_path}': {exc}") from exc
        mission = "TESS" if "Tess" in type(tpf).__name__ else "Kepler"
        catalog_id = tpf.meta.get("TICID") if mission == "TESS" else tpf.meta.get("KEPLERID")
        return tpf, mission, catalog_id

    if tic_id is not None:
        mission = "TESS"
        target = f"TIC {int(tic_id)}"
        sr = lk.search_targetpixelfile(target, mission="TESS", sector=sector)
        if len(sr) == 0:
            raise TpfResolutionError(
                f"no TESS target pixel file found for TIC {tic_id}"
                + (f" sector {sector}" if sector is not None else "")
            )
        sr = _prefer_author(sr, "SPOC")
        sr = _first_by_sequence(sr)
        catalog_id = int(tic_id)
    else:
        mission = "Kepler"
        target = f"KIC {int(kic_id)}"
        sr = lk.search_targetpixelfile(target, mission="Kepler", quarter=quarter)
        if len(sr) == 0:
            raise TpfResolutionError(
                f"no Kepler target pixel file found for KIC {kic_id}"
                + (f" quarter {quarter}" if quarter is not None else "")
            )
        sr = _prefer_author(sr, "Kepler")
        sr = _first_by_sequence(sr)
        catalog_id = int(kic_id)

    try:
        tpf = sr[0].download(download_dir=str(_cache_dir(cache_dir)))
    except Exception as exc:  # noqa: BLE001 - surface MAST/network failures uniformly
        raise TpfResolutionError(f"failed to download target pixel file for '{target}': {exc}") from exc

    return tpf, mission, catalog_id
