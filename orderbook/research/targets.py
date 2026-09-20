"""The prediction target: mid-price change over a fixed wall-clock horizon.

Rejected alternatives, and the reasoning behind the horizon, are set out in
``report.html``. In short: the micro-price is too close to the features to
be an honest target, no trade-based target is possible because the feed carries
no executions, and wall-clock beats event-count because event intensity varies
several-fold through the session.
"""

from __future__ import annotations

import numpy as np

from .features import asof_index


def forward_mid_change(
    timestamps: np.ndarray, mid: np.ndarray, horizon_us: int, tick_size: int
) -> np.ndarray:
    """Mid-price change over the next ``horizon_us``, in ticks.

    The forward value is the mid prevailing at or before ``t + horizon``, the
    last price actually observable then. Rows within one horizon of the close are
    NaN rather than being given a silently shortened horizon.

    Takes arrays rather than a frame so one mid series serves several horizons.
    """
    if horizon_us <= 0:
        raise ValueError("horizon_us must be positive")

    future_index, valid = asof_index(timestamps, horizon_us)
    future_mid = np.where(valid, mid[future_index], np.nan)
    return (future_mid - mid) / tick_size
