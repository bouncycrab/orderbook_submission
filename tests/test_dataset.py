"""Loading, subsampling policies, and dataset assembly."""

import numpy as np
import pytest
from conftest import make_book_frame

from orderbook.research.dataset import (
    book_change_mask,
    complete_book_mask,
    policy_mask,
    time_grid_mask,
    trading_day_from_path,
)


class TestTimeGrid:
    def test_selected_rows_are_at_least_one_spacing_apart(self):
        """The guarantee the whole subsampling argument rests on."""
        rng = np.random.default_rng(0)
        timestamps = np.sort(rng.integers(0, 10_000_000, size=5_000))
        spacing = 250_000

        selected = timestamps[time_grid_mask(timestamps, spacing)]

        assert (np.diff(selected) >= spacing).all()

    def test_a_dense_burst_contributes_a_single_row(self):
        mask = time_grid_mask(np.array([0, 1, 2, 3, 1_000_000]), 1_000_000)

        assert list(mask) == [True, False, False, False, True]

    def test_repeated_timestamps_do_not_stall_the_scan(self):
        assert time_grid_mask(np.zeros(100, dtype=np.int64), 1_000).sum() == 1

    def test_non_positive_spacing_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            time_grid_mask(np.array([0, 1]), 0)


class TestFilters:
    def test_book_change_flags_rows_where_the_touch_moved(self):
        frame = make_book_frame(
            timestamps_us=[0, 1, 2, 3],
            bid_prices=[10_000, 10_000, 10_000, 10_005],
            bid_sizes=[10, 10, 12, 12],
            ask_prices=[10_005, 10_005, 10_005, 10_010],
            ask_sizes=[10, 10, 10, 10],
        )

        assert list(book_change_mask(frame)) == [True, False, True, True]

    def test_complete_book_requires_every_level_on_both_sides(self):
        frame = make_book_frame(
            timestamps_us=[0, 1],
            bid_prices=[10_000, 10_000],
            bid_sizes=[10, 10],
            ask_prices=[10_005, 10_005],
            ask_sizes=[10, 10],
        )
        frame.loc[1, "aq3"] = 0

        assert list(complete_book_mask(frame, depth=5)) == [True, False]


class TestPolicyDispatch:
    def test_all_keeps_every_row(self, config):
        mask = policy_mask(np.array([0, 1, 2]), np.array([True, False, True]), config, "all")

        assert mask.all()

    def test_book_change_passes_the_touch_mask_through(self, config):
        touch_changed = np.array([True, False, True])

        mask = policy_mask(np.array([0, 1, 2]), touch_changed, config, "book_change")

        assert list(mask) == [True, False, True]

    def test_unknown_policy_is_rejected(self, config):
        with pytest.raises(ValueError, match="unknown sampling policy"):
            policy_mask(np.array([0, 1]), np.ones(2, dtype=bool), config, "nonsense")


class TestTradingDay:
    def test_day_is_taken_from_the_filename(self):
        assert trading_day_from_path("books/book_20190610.csv") == "20190610"

    def test_a_filename_without_a_date_is_rejected(self):
        with pytest.raises(ValueError, match="cannot determine trading day"):
            trading_day_from_path("books/output.csv")
