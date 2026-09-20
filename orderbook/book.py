"""The aggregate limit order book: domain types, one price ladder per side, and
the book that applies updates to them.

Prices and quantities stay integers throughout - the feed quotes whole ticks, so
integer arithmetic is exact and fast.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

from sortedcontainers import SortedDict


class Side(StrEnum):
    BID = "b"
    ASK = "a"


class Action(StrEnum):
    ADD = "a"
    DELETE = "d"
    MODIFY = "m"


class Order(NamedTuple):
    """Last known state of one resting order; side and id live in the map key."""

    price: int
    quantity: int


class Level(NamedTuple):
    """Aggregate quantity resting at one price on one side."""

    price: int
    quantity: int


class OrderUpdate(NamedTuple):
    """One row of the raw feed.

    ``order_id`` stays text because it is only ever a dictionary key. Ids are
    unique per (day, side), not globally, hence the ``(side, order_id)`` key.
    """

    timestamp: int
    side: Side
    action: Action
    order_id: str
    price: int
    quantity: int


class BookSnapshot(NamedTuple):
    """The book immediately after ``update``, best price first.

    ``bids`` and ``asks`` contain only levels that exist, so they are shorter
    than the requested depth when a side is thin; padding is the writer's job.
    """

    update: OrderUpdate
    bids: list[Level]
    asks: list[Level]


class BookIntegrityError(Exception):
    """The feed contradicts itself and the update cannot be applied.

    Raised rather than absorbed: silently skipping a bad update leaves every
    later row subtly wrong with no way to tell from the output.
    """


class PriceLadder:
    """One side of the book: price -> aggregate resting quantity.

    Prices are stored as true values, sorted ascending; only which end counts as
    "best" differs between sides. Negating bid prices so both sides sort
    best-first is the common alternative, but true prices stay readable in a
    debugger and cost only one branch when slicing the top.

    Emptied levels are deleted, not left at zero, so "best N non-zero levels" is
    a plain slice.
    """

    def __init__(self, best_is_highest: bool) -> None:
        self._best_is_highest = best_is_highest
        self._quantity_by_price: SortedDict = SortedDict()

    def add(self, price: int, quantity: int) -> None:
        self._quantity_by_price[price] = self._quantity_by_price.get(price, 0) + quantity

    def remove(self, price: int, quantity: int) -> None:
        resting = self._quantity_by_price.get(price)
        if resting is None or resting < quantity:
            raise BookIntegrityError(
                f"cannot remove {quantity} from price level {price} holding {resting or 0}"
            )
        if resting == quantity:
            del self._quantity_by_price[price]
        else:
            self._quantity_by_price[price] = resting - quantity

    def top_levels(self, depth: int) -> list[Level]:
        """Up to ``depth`` levels, best price first."""
        items = self._quantity_by_price.items()
        best_first = reversed(items[-depth:]) if self._best_is_highest else items[:depth]
        return [Level(price, quantity) for price, quantity in best_first]

    def __len__(self) -> int:
        return len(self._quantity_by_price)


class OrderBook:
    """Aggregate (level 2) book maintained update by update.

    Holds one ladder per side plus a map from ``(side, order_id)`` to each live
    order's last known state. That map is what makes modify and delete possible:
    those updates carry the order's new state but never say which price level it
    currently sits on, so the book has to remember that itself.

    No per-order time priority is modelled - the output needs only aggregate
    quantity per price, so orders within a level need no ordering.
    """

    def __init__(self) -> None:
        self._ladders = {
            Side.BID: PriceLadder(best_is_highest=True),
            Side.ASK: PriceLadder(best_is_highest=False),
        }
        self._live_orders: dict[tuple[Side, str], Order] = {}

    def apply(self, update: OrderUpdate) -> None:
        ladder = self._ladders[update.side]
        key = (update.side, update.order_id)

        if update.action == Action.ADD:
            if key in self._live_orders:
                raise BookIntegrityError(
                    f"add for {key} which is already live at t={update.timestamp}"
                )
            ladder.add(update.price, update.quantity)
            self._live_orders[key] = Order(update.price, update.quantity)

        elif update.action == Action.DELETE:
            resting = self._live_orders.pop(key, None)
            if resting is None:
                raise BookIntegrityError(
                    f"delete for unknown order {key} at t={update.timestamp}"
                )
            ladder.remove(resting.price, resting.quantity)

        elif update.action == Action.MODIFY:
            resting = self._live_orders.get(key)
            if resting is None:
                raise BookIntegrityError(
                    f"modify for unknown order {key} at t={update.timestamp}"
                )
            # The update carries the order's full new state and may move it to a
            # different price, so unwind the old state before applying the new.
            ladder.remove(resting.price, resting.quantity)
            ladder.add(update.price, update.quantity)
            self._live_orders[key] = Order(update.price, update.quantity)

        else:
            raise BookIntegrityError(
                f"unknown action {update.action!r} at t={update.timestamp}"
            )

    def top_levels(self, depth: int) -> tuple[list[Level], list[Level]]:
        return (
            self._ladders[Side.BID].top_levels(depth),
            self._ladders[Side.ASK].top_levels(depth),
        )
