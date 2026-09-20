"""Shared fixtures and helpers.

There is no ``__init__.py`` in this directory, so pytest puts it on the import
path and test modules can ``from conftest import ...`` directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from orderbook.research.config import ResearchConfig

TICK = 5
DEPTH = 5


def make_book_frame(
    timestamps_us: list[int],
    bid_prices: list[int],
    bid_sizes: list[int],
    ask_prices: list[int],
    ask_sizes: list[int],
    deeper_size: int = 10,
    day: str = "20190610",
) -> pd.DataFrame:
    """Build a frame shaped like Part 1 output.

    Levels behind the touch are filled one tick apart at a constant size, so
    completeness filters pass and depth features are defined without each test
    having to spell out twenty columns.
    """
    count = len(timestamps_us)
    columns: dict[str, object] = {
        "timestamp": timestamps_us,
        "price": bid_prices,
        "side": ["b"] * count,
        "day": [day] * count,
    }
    for level in range(DEPTH):
        columns[f"bp{level}"] = [price - level * TICK for price in bid_prices]
        columns[f"ap{level}"] = [price + level * TICK for price in ask_prices]
        columns[f"bq{level}"] = bid_sizes if level == 0 else [deeper_size] * count
        columns[f"aq{level}"] = ask_sizes if level == 0 else [deeper_size] * count
    return pd.DataFrame(columns)


def touch_frame(bid_prices, bid_sizes, ask_prices, ask_sizes) -> pd.DataFrame:
    """A book frame one second apart, described by its touch alone."""
    return make_book_frame(
        timestamps_us=[index * 1_000_000 for index in range(len(bid_prices))],
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )


@pytest.fixture
def config() -> ResearchConfig:
    """Short windows, so fixtures can stay small and readable."""
    return ResearchConfig(
        tick_size=TICK,
        depth=DEPTH,
        horizon_seconds=1.0,
        lookback_windows_seconds=(1.0,),
        warmup_seconds=0.0,
    )


@pytest.fixture
def steady_book() -> pd.DataFrame:
    """Ten updates one second apart with a one-tick spread."""
    return touch_frame([10_000] * 10, [20] * 10, [10_005] * 10, [20] * 10)
