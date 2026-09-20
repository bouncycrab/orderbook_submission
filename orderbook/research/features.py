"""Causal time-window primitives and the features built on them.

Every feature uses only information available at or before its own timestamp,
and every window is measured in wall-clock time rather than row counts: event
intensity varies several-fold through the session, so a fixed number of rows
spans very different amounts of real time at the open and mid-afternoon.

Timestamps repeat - up to 20 updates share a microsecond - so lookups resolve to
the *last* observation at a given time.

Features are computed on the full event series and only afterwards thinned to
the fitting sample, so trailing windows always see every event they should.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig


def level_columns(side: str, field: str, depth: int) -> list[str]:
    """Column names for one side and field, e.g. ``bp0..bp4``.

    ``side`` is ``"b"`` or ``"a"``; ``field`` is ``"p"`` for price, ``"q"`` for
    quantity.
    """
    return [f"{side}{field}{index}" for index in range(depth)]


def asof_index(timestamps: np.ndarray, offset_us: int) -> tuple[np.ndarray, np.ndarray]:
    """Locate the observation prevailing ``offset_us`` away from each row.

    Returns ``(index, valid)``; ``index[i]`` is the last observation at or before
    ``timestamps[i] + offset_us``. Negative offsets look back, positive ahead.
    ``valid`` is False where that time falls outside the data - at the end of the
    day there is no observation a full horizon ahead, and falling back to the
    last row would quietly shorten the horizon exactly where the session ended.
    """
    targets = timestamps + offset_us
    index = np.searchsorted(timestamps, targets, side="right") - 1
    valid = (targets >= timestamps[0]) & (targets <= timestamps[-1])
    return index, valid


def window_start_index(timestamps: np.ndarray, window_us: int) -> np.ndarray:
    """First row inside the trailing window ``[t - window_us, t]`` for each row."""
    return np.searchsorted(timestamps, timestamps - window_us, side="left")


def windowed_sum(values: np.ndarray, start_index: np.ndarray) -> np.ndarray:
    """Sum from ``start_index[i]`` to ``i`` inclusive, via a cumulative sum."""
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return cumulative[np.arange(1, values.size + 1)] - cumulative[start_index]


def windowed_mean(values: np.ndarray, start_index: np.ndarray) -> np.ndarray:
    counts = np.arange(start_index.size) - start_index + 1
    return windowed_sum(values, start_index) / counts


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide, yielding NaN where the denominator is zero.

    NaN rather than a filled value: an undefined ratio should drop out of the
    fitting sample, not be quietly imputed.
    """
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=np.float64),
        where=denominator != 0,
    )


def mid_price(frame: pd.DataFrame) -> np.ndarray:
    """Mid of the touch; NaN while either side is empty, early in the session."""
    bid = frame["bp0"].to_numpy(dtype=np.float64)
    ask = frame["ap0"].to_numpy(dtype=np.float64)
    both_present = (frame["bq0"].to_numpy() > 0) & (frame["aq0"].to_numpy() > 0)
    return np.where(both_present, (bid + ask) / 2.0, np.nan)


def order_flow_imbalance(frame: pd.DataFrame) -> np.ndarray:
    """Per-event order flow imbalance at the touch.

    Follows Cont, Kukanov and Stoikov (2014): a price move counts the whole new
    queue as arriving and the whole old queue as leaving, which separates "the
    touch grew" from "the touch moved" - something a plain change in quantity
    conflates. The first row has no predecessor and contributes zero.
    """
    bid_price = frame["bp0"].to_numpy(dtype=np.float64)
    bid_size = frame["bq0"].to_numpy(dtype=np.float64)
    ask_price = frame["ap0"].to_numpy(dtype=np.float64)
    ask_size = frame["aq0"].to_numpy(dtype=np.float64)

    previous_bid_price, previous_bid_size = np.roll(bid_price, 1), np.roll(bid_size, 1)
    previous_ask_price, previous_ask_size = np.roll(ask_price, 1), np.roll(ask_size, 1)

    bid_flow = np.where(bid_price >= previous_bid_price, bid_size, 0.0) - np.where(
        bid_price <= previous_bid_price, previous_bid_size, 0.0
    )
    ask_flow = np.where(ask_price <= previous_ask_price, ask_size, 0.0) - np.where(
        ask_price >= previous_ask_price, previous_ask_size, 0.0
    )

    flow = bid_flow - ask_flow
    flow[0] = 0.0
    return flow


def build_features(frame: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Build the feature matrix for one day, aligned row-for-row with ``frame``.

    Rows where a feature is undefined carry NaN and are dropped when the fitting
    sample is assembled.
    """
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    tick = float(config.tick_size)

    bid_size = frame["bq0"].to_numpy(dtype=np.float64)
    ask_size = frame["aq0"].to_numpy(dtype=np.float64)
    touch_size = bid_size + ask_size

    mid = mid_price(frame)
    spread_ticks = (
        frame["ap0"].to_numpy(dtype=np.float64) - frame["bp0"].to_numpy(dtype=np.float64)
    ) / tick
    spread_ticks = np.where(np.isnan(mid), np.nan, spread_ticks)

    imbalance = safe_divide(bid_size - ask_size, touch_size)
    total_bid = frame[level_columns("b", "q", config.depth)].to_numpy(float).sum(axis=1)
    total_ask = frame[level_columns("a", "q", config.depth)].to_numpy(float).sum(axis=1)

    features: dict[str, np.ndarray] = {
        "spread_ticks": spread_ticks,
        # Queue imbalance at the touch: the main thing that varies in a book
        # that is one tick wide almost all the time.
        #
        # The size-weighted micro-price offset was also tried here. It is
        # algebraically (spread / 2) * imbalance, and since the spread is one
        # tick for most of the session the two correlate at 0.99 (VIF above 50).
        # Carrying both bought 0.001 of out-of-sample IC, so only the more
        # interpretable of the pair is kept.
        "imbalance_touch": imbalance,
        # How much of each side sits at the front: a queue with depth behind it
        # behaves differently from one that is all at the touch.
        "touch_share_bid": safe_divide(bid_size, total_bid),
        "touch_share_ask": safe_divide(ask_size, total_ask),
    }

    flow = order_flow_imbalance(frame)

    for window_seconds in config.lookback_windows_seconds:
        window_us = config.window_us(window_seconds)
        label = f"{window_seconds:g}s"
        start = window_start_index(timestamps, window_us)

        # Drift against the mid prevailing a window ago, not the mid of whichever
        # row happens to sit that many positions back.
        past_index, past_valid = asof_index(timestamps, -window_us)
        features[f"mid_change_{label}"] = (
            mid - np.where(past_valid, mid[past_index], np.nan)
        ) / tick

        # Scaled by typical touch size over the same window, so the feature means
        # "flow relative to queue size" and stays comparable across the session.
        features[f"ofi_{label}"] = safe_divide(
            windowed_sum(flow, start), windowed_mean(touch_size, start)
        )

    return pd.DataFrame(features, index=frame.index)
