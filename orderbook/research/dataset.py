"""Loading Part 1 output, choosing which rows to fit on, and assembling samples.

**The day is the unit of work.** The book is rebuilt from empty each morning and
fourteen hours of closed market separate the files, so windows and forward
targets never span two days. Days are concatenated only at the end, keeping a
day label so validation can split on it.

**The data is subsampled, deliberately.** Part 1 emits one row per update -
about 240,000 on a busy day - and consecutive rows are near-duplicates, since
most updates move neither the touch price nor its size. Worse, a five second
forward target is shared almost in full by every row inside any five second
stretch. Fitting on all of them gives tens of thousands of overlapping views of
a few thousand independent events: the model is not helped, it simply becomes
confident about a sample far smaller than its row count, and every standard
error and validation score computed from it is optimistic.

The fitting sample is therefore thinned to rows at least one horizon apart, so
consecutive observations have non-overlapping forward windows. Two alternative
policies are available from the command line for comparison: ``all`` keeps every
row, and ``book_change`` keeps rows where the touch moved. Both leave
overlapping targets, so neither is the default.

**Features are computed before thinning, never after.** Trailing windows need
every event; only the rows used for *fitting* are thinned.

The work splits in two so a different horizon costs little: :func:`build_panels`
does the expensive per-day work once, :func:`assemble_dataset` is cheap.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .features import build_features, level_columns, mid_price
from .targets import forward_mid_change

SamplingPolicy = Literal["time_grid", "book_change", "all"]

TOUCH_COLUMNS = ("bp0", "bq0", "ap0", "aq0")

_DAY_PATTERN = re.compile(r"(\d{8})")


def trading_day_from_path(path: str | Path) -> str:
    """Extract the ``YYYYMMDD`` trading day from a book filename."""
    match = _DAY_PATTERN.search(Path(path).stem)
    if match is None:
        raise ValueError(
            f"cannot determine trading day from {Path(path).name!r}; "
            "expected a YYYYMMDD date in the filename"
        )
    return match.group(1)


def load_book_day(path: str | Path, depth: int = 5) -> pd.DataFrame:
    """Load one day of book state, tagged with its trading day."""
    path = Path(path)
    required = ["timestamp", "price", "side"] + [
        column
        for side in ("b", "a")
        for field in ("p", "q")
        for column in level_columns(side, field, depth)
    ]

    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps are not non-decreasing")

    frame["day"] = trading_day_from_path(path)
    return frame


def load_book_days(paths: Iterable[str | Path], depth: int = 5) -> Iterator[pd.DataFrame]:
    """Load each path in chronological order, one frame per day."""
    for path in sorted(paths, key=trading_day_from_path):
        yield load_book_day(path, depth)


def time_grid_mask(timestamps: np.ndarray, spacing_us: int) -> np.ndarray:
    """Select rows at least ``spacing_us`` apart, scanning forwards.

    A greedy scan rather than bucketing on ``timestamp // spacing``: bucketing
    gives one row per bucket but no minimum gap between the rows chosen, and two
    selections either side of a bucket edge can be microseconds apart - exactly
    the overlap the thinning exists to remove.
    """
    if spacing_us <= 0:
        raise ValueError("spacing_us must be positive")

    mask = np.zeros(timestamps.size, dtype=bool)
    position = 0
    while position < timestamps.size:
        mask[position] = True
        position = int(np.searchsorted(timestamps, timestamps[position] + spacing_us))
    return mask


def book_change_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select rows where the touch price or size changed from the row before."""
    touch = frame.loc[:, list(TOUCH_COLUMNS)].to_numpy()
    changed = np.ones(len(frame), dtype=bool)
    changed[1:] = np.any(touch[1:] != touch[:-1], axis=1)
    return changed


def complete_book_mask(frame: pd.DataFrame, depth: int) -> np.ndarray:
    """Require every level on both sides; depth features need a full book."""
    columns = level_columns("b", "q", depth) + level_columns("a", "q", depth)
    return (frame.loc[:, columns].to_numpy() > 0).all(axis=1)


def policy_mask(
    timestamps: np.ndarray,
    touch_changed: np.ndarray,
    config: ResearchConfig,
    policy: SamplingPolicy,
) -> np.ndarray:
    """Apply one sampling policy to a prepared panel."""
    if policy == "time_grid":
        return time_grid_mask(timestamps, config.spacing_us)
    if policy == "book_change":
        return touch_changed
    if policy == "all":
        return np.ones(timestamps.size, dtype=bool)
    raise ValueError(f"unknown sampling policy {policy!r}")


@dataclass(frozen=True)
class DayPanel:
    """One day of full-resolution features, ready to be sampled from.

    The raw frame is not retained: once features, mid and the horizon-independent
    filters are computed, nothing downstream needs it.
    """

    day: str
    timestamps: np.ndarray
    mid: np.ndarray
    features: pd.DataFrame
    eligible: np.ndarray
    touch_changed: np.ndarray

    def __len__(self) -> int:
        return self.timestamps.size


@dataclass(frozen=True)
class Dataset:
    """A fitting sample: features, target, and the day each row came from."""

    features: pd.DataFrame
    target: pd.Series
    day: pd.Series
    timestamp: pd.Series
    horizon_seconds: float

    def __len__(self) -> int:
        return len(self.features)

    @property
    def days(self) -> list[str]:
        return sorted(self.day.unique())

    def rows_for_days(self, days: Sequence[str]) -> "Dataset":
        mask = self.day.isin(list(days)).to_numpy()
        return Dataset(
            features=self.features.loc[mask],
            target=self.target.loc[mask],
            day=self.day.loc[mask],
            timestamp=self.timestamp.loc[mask],
            horizon_seconds=self.horizon_seconds,
        )

    def rows_per_day(self) -> pd.Series:
        return self.day.value_counts().sort_index()

    def to_frame(self) -> pd.DataFrame:
        """One tidy frame: identifiers, every feature, then the target.

        This is the modelling sample exactly as the model sees it, which makes
        it inspectable outside the pipeline - sorted, filtered or plotted in a
        spreadsheet without having to trust the code that built it.
        """
        return pd.concat(
            [
                self.day.reset_index(drop=True),
                self.timestamp.reset_index(drop=True),
                self.features.reset_index(drop=True),
                self.target.reset_index(drop=True),
            ],
            axis=1,
        )


def build_panel(frame: pd.DataFrame, config: ResearchConfig) -> DayPanel:
    """Compute features and the horizon-independent filters for one day.

    Rows are excluded here if they fall in the warm-up while the book is still
    filling in from empty, if a level is missing on either side, or if any
    feature is undefined.
    """
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    features = build_features(frame, config)

    eligible = (
        (timestamps >= timestamps[0] + config.warmup_us)
        & complete_book_mask(frame, config.depth)
        & np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    )

    return DayPanel(
        day=str(frame["day"].iloc[0]),
        timestamps=timestamps,
        mid=mid_price(frame),
        features=features,
        eligible=eligible,
        touch_changed=book_change_mask(frame),
    )


def build_panels(paths: Iterable[str | Path], config: ResearchConfig) -> list[DayPanel]:
    """Load and prepare every day, in chronological order."""
    return [build_panel(frame, config) for frame in load_book_days(paths, config.depth)]


def assemble_dataset(
    panels: Sequence[DayPanel],
    config: ResearchConfig,
    policy: SamplingPolicy = "time_grid",
) -> Dataset:
    """Build a fitting sample from prepared panels at the configured horizon."""
    features_blocks, targets, days, stamps = [], [], [], []

    for panel in panels:
        target = forward_mid_change(
            panel.timestamps, panel.mid, config.horizon_us, config.tick_size
        )
        selected = (
            panel.eligible
            & policy_mask(panel.timestamps, panel.touch_changed, config, policy)
            & np.isfinite(target)
        )

        features_blocks.append(panel.features.loc[selected])
        targets.append(target[selected])
        days.append(np.full(int(selected.sum()), panel.day))
        stamps.append(panel.timestamps[selected])

    return Dataset(
        features=pd.concat(features_blocks, ignore_index=True),
        target=pd.Series(np.concatenate(targets), name="forward_mid_change_ticks"),
        day=pd.Series(np.concatenate(days), name="day"),
        timestamp=pd.Series(np.concatenate(stamps), name="timestamp"),
        horizon_seconds=config.horizon_seconds,
    )
