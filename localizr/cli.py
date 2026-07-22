"""Command-line interface for localizr."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import LocalizationError, __version__, localize
from .diffimage import DifferenceImageError
from .gaia import GaiaQueryError
from .tpf import TpfResolutionError

_VERDICT_COLOR = {"on_target": "green", "off_target_blend": "red", "inconclusive": "yellow"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localizr",
        description="Pixel-level transit localization via TESS/Kepler TPF difference imaging.",
    )
    parser.add_argument("--version", action="version", version=f"localizr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    localize_p = sub.add_parser(
        "localize", help="Localize a transit signal against its target pixel file"
    )

    target = localize_p.add_argument_group("target")
    target.add_argument("--tic-id", type=int, default=None, dest="tic_id", metavar="ID")
    target.add_argument("--kic-id", type=int, default=None, dest="kic_id", metavar="ID")
    target.add_argument("--tpf-path", type=str, default=None, dest="tpf_path", metavar="PATH")
    target.add_argument("--sector", type=int, default=None, help="TESS sector (default: first available)")
    target.add_argument("--quarter", type=int, default=None, help="Kepler quarter (default: first available)")

    ephemeris = localize_p.add_argument_group("ephemeris")
    ephemeris.add_argument("--period", type=float, required=True, metavar="DAYS")
    ephemeris.add_argument(
        "--epoch",
        type=float,
        required=True,
        dest="epoch_btjd",
        metavar="TIME",
        help="Transit epoch, same time system as the TPF's own TIME column "
        "(BTJD for TESS, BKJD for Kepler)",
    )
    ephemeris.add_argument("--duration-hours", type=float, required=True, dest="duration_hours")

    output = localize_p.add_argument_group("output")
    output.add_argument("--plot", action="store_true", default=False)
    output.add_argument("--plot-path", type=str, default=None, dest="plot_path", metavar="PATH")
    output.add_argument("--json", action="store_true", dest="json_output")
    output.add_argument("--json-path", type=str, default=None, dest="json_path", metavar="PATH")
    output.add_argument("--cache-dir", type=str, default=None, dest="cache_dir", metavar="DIR")

    return parser


def _run_localize(args: argparse.Namespace, console: Console) -> int:
    try:
        result = localize(
            tic_id=args.tic_id,
            kic_id=args.kic_id,
            tpf_path=args.tpf_path,
            period=args.period,
            epoch_btjd=args.epoch_btjd,
            duration_hours=args.duration_hours,
            sector=args.sector,
            quarter=args.quarter,
            cache_dir=args.cache_dir,
        )
    except (TpfResolutionError, DifferenceImageError, GaiaQueryError, LocalizationError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2

    plot_path = None
    if args.plot:
        label = result.tic_id if result.mission == "TESS" else result.kic_id
        default_name = f"{result.mission.lower()}_{label}_localizr.png"
        out_path = Path(args.plot_path) if args.plot_path else Path.cwd() / default_name
        try:
            plot_path = result.save_plot(str(out_path))
        except LocalizationError as exc:
            console.print(f"[yellow]Warning:[/yellow] could not save plot: {exc}")

    if args.json_output:
        print(result.to_json(args.json_path))
    else:
        _print_result(result, console, plot_path)

    return 1 if result.verdict == "off_target_blend" else 0


def _print_result(result, console: Console, plot_path: str | None) -> None:
    color = _VERDICT_COLOR[result.verdict]
    label = result.tic_id if result.mission == "TESS" else result.kic_id
    id_kind = "TIC" if result.mission == "TESS" else "KIC"

    console.print(f"[bold]{id_kind} {label}[/bold] ({result.mission})")
    console.print(f"Verdict: [{color}]{result.verdict}[/{color}]")
    console.print(result.verdict_reason)

    if result.centroid_offset_arcsec is not None:
        sigma_str = (
            f"{result.centroid_offset_sigma:.2f}"
            if result.centroid_offset_sigma is not None
            else "n/a"
        )
        console.print(
            f"Centroid offset: {result.centroid_offset_arcsec:.3f}\" ({sigma_str} sigma)"
        )

    if result.nearby_gaia_sources:
        console.print(f"Nearby Gaia DR3 sources: {len(result.nearby_gaia_sources)}")
        for src in result.nearby_gaia_sources[:5]:
            gmag = f"{src['gmag']:.2f}" if src["gmag"] is not None else "n/a"
            console.print(f"  - {src['source_id']}  sep={src['sep_arcsec']:.2f}\"  G={gmag}")

    if plot_path:
        console.print(f"Saved plot: {plot_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.command == "localize":
        return _run_localize(args, console)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
