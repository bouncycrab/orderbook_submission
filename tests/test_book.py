"""The price ladder and the book itself."""

import pytest

from orderbook.book import (
    Action,
    BookIntegrityError,
    Level,
    OrderBook,
    OrderUpdate,
    PriceLadder,
    Side,
)


def update(action, side, order_id, price, quantity, timestamp=0) -> OrderUpdate:
    return OrderUpdate(timestamp, Side(side), Action(action), order_id, price, quantity)


class TestPriceLadder:
    def test_add_accumulates_quantity_at_the_same_price(self):
        ladder = PriceLadder(best_is_highest=True)
        ladder.add(100, 3)
        ladder.add(100, 5)

        assert ladder.top_levels(5) == [Level(100, 8)]

    def test_partial_remove_leaves_the_level_in_place(self):
        ladder = PriceLadder(best_is_highest=True)
        ladder.add(100, 8)
        ladder.remove(100, 3)

        assert ladder.top_levels(5) == [Level(100, 5)]

    def test_fully_consumed_level_is_dropped_not_zeroed(self):
        ladder = PriceLadder(best_is_highest=True)
        ladder.add(100, 3)
        ladder.add(95, 4)
        ladder.remove(100, 3)

        assert ladder.top_levels(5) == [Level(95, 4)]
        assert len(ladder) == 1

    def test_each_side_orders_from_its_own_best_price(self):
        bids = PriceLadder(best_is_highest=True)
        asks = PriceLadder(best_is_highest=False)
        for price in (95, 105, 100):
            bids.add(price, 1)
            asks.add(price, 1)

        assert [level.price for level in bids.top_levels(5)] == [105, 100, 95]
        assert [level.price for level in asks.top_levels(5)] == [95, 100, 105]

    def test_top_levels_truncates_to_depth_and_tolerates_a_thin_side(self):
        ladder = PriceLadder(best_is_highest=False)
        for price in range(100, 110):
            ladder.add(price, 1)

        assert [level.price for level in ladder.top_levels(3)] == [100, 101, 102]
        assert len(PriceLadder(best_is_highest=False).top_levels(5)) == 0

    @pytest.mark.parametrize("resting,removing", [(3, 4), (0, 1)])
    def test_removing_more_than_rests_is_an_integrity_error(self, resting, removing):
        ladder = PriceLadder(best_is_highest=True)
        if resting:
            ladder.add(100, resting)

        with pytest.raises(BookIntegrityError):
            ladder.remove(100, removing)


def test_specification_worked_example():
    """The five-step scenario from the assessment document, in order."""
    book = OrderBook()

    # 1. Add id=1 bid 15 x3 -> a single level.
    book.apply(update("a", "b", "1", 15, 3))
    assert book.top_levels(5) == ([Level(15, 3)], [])

    # 2. Add id=2 at the same price -> quantity aggregates to 8.
    book.apply(update("a", "b", "2", 15, 5))
    assert book.top_levels(5) == ([Level(15, 8)], [])

    # 3. Modify id=1 to price 20 -> the order moves, leaving 5 behind at 15.
    book.apply(update("m", "b", "1", 20, 3))
    assert book.top_levels(5) == ([Level(20, 3), Level(15, 5)], [])

    # 4. Add on the far side -> bids untouched.
    book.apply(update("a", "a", "3", 30, 1))
    assert book.top_levels(5) == ([Level(20, 3), Level(15, 5)], [Level(30, 1)])

    # 5. Delete id=2 -> the 15 level empties and disappears entirely.
    book.apply(update("d", "b", "2", 15, 5))
    assert book.top_levels(5) == ([Level(20, 3)], [Level(30, 1)])


def test_order_ids_are_scoped_per_side_not_globally():
    """Ids are unique per (day, side), so one id can be live on both sides."""
    book = OrderBook()
    book.apply(update("a", "b", "7", 100, 4))
    book.apply(update("a", "a", "7", 110, 9))
    book.apply(update("d", "b", "7", 100, 4))

    assert book.top_levels(5) == ([], [Level(110, 9)])


def test_modify_that_only_changes_quantity_keeps_the_level():
    book = OrderBook()
    book.apply(update("a", "b", "1", 100, 10))
    book.apply(update("a", "b", "2", 100, 5))
    book.apply(update("m", "b", "1", 100, 2))

    assert book.top_levels(5)[0] == [Level(100, 7)]


def test_delete_uses_remembered_state_not_the_row_contents():
    """The delete row here claims a stale price; the real level must be reduced.

    The sample feed happens to echo the order's true state back on deletes, but
    the book must not depend on that.
    """
    book = OrderBook()
    book.apply(update("a", "b", "1", 100, 10))
    book.apply(update("m", "b", "1", 105, 4))
    book.apply(update("d", "b", "1", 100, 10))

    assert book.top_levels(5)[0] == []


class TestFeedIntegrity:
    def test_delete_for_unknown_order_raises(self):
        with pytest.raises(BookIntegrityError, match="unknown order"):
            OrderBook().apply(update("d", "b", "1", 100, 1))

    def test_modify_for_unknown_order_raises(self):
        with pytest.raises(BookIntegrityError, match="unknown order"):
            OrderBook().apply(update("m", "b", "1", 100, 1))

    def test_add_reusing_a_live_id_raises(self):
        book = OrderBook()
        book.apply(update("a", "b", "1", 100, 1))

        with pytest.raises(BookIntegrityError, match="already live"):
            book.apply(update("a", "b", "1", 100, 1))

    def test_id_may_be_reused_once_the_order_is_deleted(self):
        book = OrderBook()
        book.apply(update("a", "b", "1", 100, 1))
        book.apply(update("d", "b", "1", 100, 1))
        book.apply(update("a", "b", "1", 105, 2))

        assert book.top_levels(5)[0] == [Level(105, 2)]
