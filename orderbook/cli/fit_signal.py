"""Part 2 entry point: build features, fit the signal, report how it did.

    python -m orderbook.cli.fit_signal books/book_*.csv

Prints a report rather than writing artefacts: the point of the exercise is the
reasoning and the measured result, and a fitted linear model on ten features
is small enough to read directly. ``--export`` writes the fitting sample out if
you want to inspect it yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..research.config import ResearchConfig
from ..research.dataset import assemble_dataset, build_panels
from ..research.model import SignalModel, ols_summary, scan_horizons, walk_forward


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a predictive order-book signal from reconstructed books.",
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="book CSVs produced by build_books"
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=ResearchConfig.horizon_seconds,
        help="prediction horizon in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--sampling",
        choices=("time_grid", "book_change", "all"),
        default="time_grid",
        help="which rows to fit on (default: %(default)s)",
    )
    parser.add_argument(
        "--scan-horizons",
        action="store_true",
        help="compare candidate horizons before reporting the chosen one",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print an OLS fit with standard errors and 95%% confidence intervals",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="write the fitting sample (features and target) to this CSV",
    )
    return parser.parse_args(argv)


def print_section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ResearchConfig(horizon_seconds=args.horizon)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(f"Loading {len(args.inputs)} day(s) and building features...")
        panels = build_panels(args.inputs, config)
        total_events = sum(len(panel) for panel in panels)
        print(f"{total_events:,} book updates across {len(panels)} day(s)")

        if args.scan_horizons:
            print_section("Horizon scan (pooled out-of-sample)")
            print(scan_horizons(panels, config, args.sampling).round(4))

        dataset = assemble_dataset(panels, config, args.sampling)
        print_section(
            f"Fitting sample: horizon {config.horizon_seconds:g}s, "
            f"sampling '{args.sampling}'"
        )
        print(f"{len(dataset):,} rows kept from {total_events:,} updates "
              f"({len(dataset) / total_events:.2%})")
        print(f"target std: {dataset.target.std():.4f} ticks")
        print("rows per day:")
        print(dataset.rows_per_day().to_string())

        if args.summary:
            print_section("OLS fit with inference (standardised features, ticks per sd)")
            print(ols_summary(dataset).summary())

        result = walk_forward(dataset, config)

        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            # Attach the out-of-sample prediction to each row. The first trading
            # day is training data only, so its rows carry no prediction.
            export = dataset.to_frame().merge(
                result.predictions_frame[["day", "timestamp", "predicted_ticks"]],
                on=["day", "timestamp"],
                how="left",
            )
            export.to_csv(args.export, index=False)
            print(f"\nfitting sample written to {args.export} ({len(export):,} rows, "
                  f"{export['predicted_ticks'].notna().sum():,} with predictions)")

        print_section("Walk-forward validation (train on prior days, test on next)")
        print(result.metrics_frame.round(4).to_string())

        print_section("Pooled out-of-sample")
        pooled = result.pooled_metrics
        for name, value in pooled.items():
            print(f"  {name:28s} {value: .4f}")

        print_section("Mean standardised coefficients (ticks per feature sd)")
        print(result.mean_coefficients.round(4).to_string())

        print_section("Predicted against realised, 20 equal-count bins")
        print(result.calibration(20).round(4).to_string(index=False))

        print_section("Realised move by predicted decile (pooled out-of-sample)")
        print(result.pooled_bucket_profile(config.profile_buckets).round(4).to_string(index=False))

        final = SignalModel.from_config(config).fit(dataset.features, dataset.target)
        print_section("Final model refitted on all days")
        print(f"  ridge alpha: {final.alpha:g}")
        print(f"  intercept:   {final.intercept: .5f} ticks")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
