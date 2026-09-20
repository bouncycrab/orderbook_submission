"""Aggregate order book reconstruction and order-book signal research.

Two halves matching the two parts of the assessment:

* :mod:`orderbook.book` and :mod:`orderbook.io` - rebuilding the aggregate book
  from raw order updates;
* :mod:`orderbook.research` - features, targets and models built on that output.

The book carries no file handling, so Part 2 can drive it directly when it needs
state the Part 1 CSV does not carry.
"""

from .book import (
    Action,
    BookIntegrityError,
    BookSnapshot,
    Level,
    Order,
    OrderBook,
    OrderUpdate,
    PriceLadder,
    Side,
)
from .io import (
    DEFAULT_DEPTH,
    output_columns,
    read_updates,
    replay,
    replay_file,
)

__all__ = [
    "Action",
    "BookIntegrityError",
    "BookSnapshot",
    "DEFAULT_DEPTH",
    "Level",
    "Order",
    "OrderBook",
    "OrderUpdate",
    "PriceLadder",
    "Side",
    "output_columns",
    "read_updates",
    "replay",
    "replay_file",
]
