"""Windowing primitives, the features built on them, and the target."""

import numpy as np
import pandas as pd
import pytest
from conftest import TICK, touch_frame

from orderbook.research.features import (
    asof_index,
    build_features,
    mid_price,
    order_flow_imbalance,
    safe_divide,
    window_start_index,
    windowed_sum,
)
from orderbook.research.targets import forward_mid_change


class TestWindows:
    def test_forward_lookup_finds_the_last_observation_within_the_horizon(self):
        index, valid = asof_index(np.array([0, 10, 20, 30, 40]), 20)

        assert list(index[:3]) == [2, 3, 4]
        assert list(valid) == [True, True, True, False, False]

    def test_backward_lookup_finds_the_last_observation_at_or_before(self):
        index, valid = asof_index(np.array([0, 10, 20, 30, 40]), -20)

        assert list(index[2:]) == [0, 1, 2]
        assert list(valid) == [False, False, True, True, True]

    def test_ties_resolve_to_the_last_observation_at_that_instant(self):
        index, _ = asof_index(np.array([0, 10, 10, 10, 20]), 10)

        assert index[0] == 3

    def test_trailing_window_includes_its_boundary(self):
        assert list(window_start_index(np.array([0, 10, 20, 30]), 20)) == [0, 0, 0, 1]

    def test_windowed_sum_is_inclusive_of_the_current_row(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])

        assert list(windowed_sum(values, np.array([0, 0, 1, 2]))) == [1.0, 3.0, 5.0, 7.0]

    def test_windowed_sums_only_ever_look_backwards(self):
        values, timestamps = np.arange(10.0), np.arange(10) * 10
        full = windowed_sum(values, window_start_index(timestamps, 30))
        cut = windowed_sum(values[:6], window_start_index(timestamps[:6], 30))

        assert full[:6] == pytest.approx(cut)

    def test_divide_by_zero_gives_nan_not_infinity(self):
        result = safe_divide(np.array([1.0, 2.0]), np.array([0.0, 2.0]))

        assert np.isnan(result[0])
        assert result[1] == pytest.approx(1.0)


class TestMidAndFlow:
    def test_mid_is_the_average_of_the_touch(self):
        assert mid_price(touch_frame([10_000], [5], [10_005], [5]))[0] == pytest.approx(10_002.5)

    def test_mid_is_undefined_when_a_side_is_empty(self):
        frame = touch_frame([10_000], [5], [10_005], [5])
        frame.loc[0, "aq0"] = 0

        assert np.isnan(mid_price(frame)[0])

    def test_first_row_contributes_no_flow(self):
        frame = touch_frame([10_000, 10_000], [10, 10], [10_005, 10_005], [10, 10])

        assert order_flow_imbalance(frame)[0] == 0.0

    @pytest.mark.parametrize(
        "bids,bid_sizes,asks,ask_sizes,expected",
        [
            # A growing bid queue is buying pressure, a growing ask queue selling.
            ([10_000, 10_000], [10, 15], [10_010, 10_010], [10, 10], 5.0),
            ([10_000, 10_000], [10, 10], [10_010, 10_010], [10, 15], -5.0),
            # A price improvement counts the whole new queue as arriving...
            ([10_000, 10_005], [10, 7], [10_010, 10_010], [10, 10], 7.0),
            # ...and a step down counts the whole old queue as leaving.
            ([10_000, 9_995], [10, 8], [10_010, 10_010], [10, 10], -10.0),
            # Symmetric moves on both sides cancel.
            ([10_000, 10_000], [10, 15], [10_010, 10_010], [10, 15], 0.0),
        ],
    )
    def test_order_flow_imbalance_cases(self, bids, bid_sizes, asks, ask_sizes, expected):
        frame = touch_frame(bids, bid_sizes, asks, ask_sizes)

        assert order_flow_imbalance(frame)[1] == pytest.approx(expected)


class TestBookShapeFeatures:
    def test_imbalance_is_positive_when_bids_outweigh_asks(self, config):
        features = build_features(touch_frame([10_000], [30], [10_005], [10]), config)

        assert features.loc[0, "imbalance_touch"] == pytest.approx(0.5)

    def test_spread_is_reported_in_ticks(self, config):
        features = build_features(touch_frame([10_000], [10], [10_010], [10]), config)

        assert features.loc[0, "spread_ticks"] == pytest.approx(2.0)

    def test_touch_share_reflects_concentration_at_the_front(self, config):
        # A touch of 60 with four deeper levels of 10 each: 60 / 100.
        features = build_features(touch_frame([10_000], [60], [10_005], [10]), config)

        assert features.loc[0, "touch_share_bid"] == pytest.approx(0.6)


def test_features_never_use_information_from_later_rows(config, steady_book):
    """Truncating the data must leave every earlier feature value unchanged.

    This is the property that makes the whole exercise meaningful, so it is
    asserted directly rather than inferred from reading the code.
    """
    full = build_features(steady_book, config)
    truncated = build_features(steady_book.iloc[:6].copy(), config)

    pd.testing.assert_frame_equal(full.iloc[:6], truncated)


class TestTarget:
    def test_forward_change_is_measured_in_ticks(self):
        timestamps = np.array([0, 1_000_000, 2_000_000])
        mid = np.array([10_000.0, 10_005.0, 10_010.0])

        target = forward_mid_change(timestamps, mid, 1_000_000, TICK)

        assert target[0] == pytest.approx(1.0)

    def test_the_mid_prevailing_at_the_horizon_is_used(self):
        """With no update exactly one second out, the last earlier mid applies."""
        timestamps = np.array([0, 400_000, 900_000, 3_000_000])
        mid = np.array([10_000.0, 10_005.0, 10_010.0, 10_100.0])

        target = forward_mid_change(timestamps, mid, 1_000_000, TICK)

        assert target[0] == pytest.approx(2.0)

    def test_rows_without_a_full_horizon_ahead_are_nan(self):
        timestamps = np.array([0, 1_000_000, 1_500_000])
        mid = np.array([10_000.0, 10_005.0, 10_010.0])

        target = forward_mid_change(timestamps, mid, 1_000_000, TICK)

        assert not np.isnan(target[0])
        assert np.isnan(target[1:]).all()

    def test_a_flat_market_has_a_zero_target(self):
        target = forward_mid_change(np.arange(5) * 1_000_000, np.full(5, 10_000.0), 1_000_000, TICK)

        assert target[:4] == pytest.approx(np.zeros(4))

    def test_non_positive_horizon_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            forward_mid_change(np.array([0, 1]), np.array([1.0, 2.0]), 0, TICK)
