"""Part 1 entry point: reconstruct the aggregate book for one or more days.

    python -m orderbook.cli.build_books data/res_*.csv --output-dir books

Each input file is one day and is processed independently, starting from an
empty book.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..book import BookIntegrityError
from ..io import DEFAULT_DEPTH, replay_file

#: Input files are named ``res_<date>.csv``; outputs are named after the date
#: alone so that downstream code can recover the trading day from the filename.
INPUT_PREFIX = "res_"
OUTPUT_PREFIX = "book_"


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    """Map an input file to its output file, preserving the date in the name."""
    stem = input_path.stem
    if stem.startswith(INPUT_PREFIX):
        stem = stem[len(INPUT_PREFIX) :]
    return output_dir / f"{OUTPUT_PREFIX}{stem}.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct aggregate order books from raw order updates.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="raw update CSVs, one per trading day",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("books"),
        help="directory for reconstructed book CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="price levels per side to output (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.depth < 1:
        print("error: --depth must be at least 1", file=sys.stderr)
        return 2

    for input_path in args.inputs:
        if not input_path.is_file():
            print(f"error: no such file: {input_path}", file=sys.stderr)
            return 2

        destination = output_path_for(input_path, args.output_dir)
        started = time.perf_counter()
        try:
            rows = replay_file(input_path, destination, args.depth)
        except (BookIntegrityError, ValueError) as error:
            print(f"error: {input_path}: {error}", file=sys.stderr)
            return 1

        elapsed = time.perf_counter() - started
        print(f"{input_path} -> {destination}  {rows:,} rows in {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
