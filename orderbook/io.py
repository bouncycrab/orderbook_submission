"""Turning files into books and back: parsing the feed, driving the book over a
day, checking the result, and writing book snapshots out.

Kept apart from :mod:`orderbook.book` so the book itself stays a pure in-memory
structure that Part 2 can drive directly.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

from .book import Action, BookSnapshot, Level, OrderBook, OrderUpdate, Side

INPUT_COLUMNS = ("timestamp", "side", "action", "id", "price", "quantity")

#: Price written for a level that does not exist. The specification ignores the
#: price of empty levels, so any value works; 0 is unmistakably not a real price.
EMPTY_LEVEL_PRICE = 0

#: Price levels per side required by the specification.
DEFAULT_DEPTH = 5

_EMPTY_LEVEL = Level(EMPTY_LEVEL_PRICE, 0)

# Direct dict lookups rather than Side(value): same members, but bypassing the
# Enum call saves meaningful time over the ~1.2M rows in the sample data.
_SIDES = {member.value: member for member in Side}
_ACTIONS = {member.value: member for member in Action}


def read_updates(path: str | Path) -> Iterator[OrderUpdate]:
    """Stream ``path`` as typed records, lazily - the book only needs one row."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: file is empty")
        if tuple(header) != INPUT_COLUMNS:
            raise ValueError(
                f"{path}: unexpected header {header}, expected {list(INPUT_COLUMNS)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp, side, action, order_id, price, quantity = row
                yield OrderUpdate(
                    timestamp=int(timestamp),
                    side=_SIDES[side],
                    action=_ACTIONS[action],
                    order_id=order_id,
                    price=int(price),
                    quantity=int(quantity),
                )
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: malformed row {row}") from error


def output_columns(depth: int) -> list[str]:
    """Header for the Part 1 output, named as the specification requires."""
    columns = ["timestamp", "price", "side"]
    for prefix in ("b", "a"):
        for index in range(depth):
            columns += [f"{prefix}p{index}", f"{prefix}q{index}"]
    return columns


def snapshot_row(snapshot: BookSnapshot, depth: int) -> list[object]:
    """Flatten a snapshot into a row, padding levels the book does not have."""
    update = snapshot.update
    row: list[object] = [update.timestamp, update.price, update.side.value]
    for levels in (snapshot.bids, snapshot.asks):
        for index in range(depth):
            level = levels[index] if index < len(levels) else _EMPTY_LEVEL
            row += [level.price, level.quantity]
    return row


def replay(
    updates: Iterable[OrderUpdate], depth: int = DEFAULT_DEPTH
) -> Iterator[BookSnapshot]:
    """Apply each update in turn, yielding the resulting top of book.

    One snapshot per update, including updates that leave the top levels
    untouched, because the specification requires one output row per input row.
    """
    book = OrderBook()
    for update in updates:
        book.apply(update)
        bids, asks = book.top_levels(depth)
        yield BookSnapshot(update, bids, asks)


def replay_file(
    input_path: str | Path, output_path: str | Path, depth: int = DEFAULT_DEPTH
) -> int:
    """Reconstruct one day into ``output_path``; returns rows written.

    Streams throughout, so memory use is independent of file size. Each file is
    one day and starts from an empty book.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(output_columns(depth))
        for snapshot in replay(read_updates(input_path), depth):
            writer.writerow(snapshot_row(snapshot, depth))
            rows_written += 1
    return rows_written
