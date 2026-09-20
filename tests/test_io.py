"""Parsing the feed, writing snapshots, and the end-to-end Part 1 pipeline."""

import csv

import pytest

from orderbook.book import Action, BookSnapshot, Level, OrderUpdate, Side
from orderbook.io import (
    EMPTY_LEVEL_PRICE,
    output_columns,
    read_updates,
    replay_file,
    snapshot_row,
)

HEADER = "timestamp,side,action,id,price,quantity\n"

# The worked example from the assessment document, as an input file.
WORKED_EXAMPLE = HEADER + "1,b,a,1,15,3\n2,b,a,2,15,5\n3,b,m,1,20,3\n4,a,a,3,30,1\n5,b,d,2,15,5\n"


def write_csv(path, body: str):
    path.write_text(body)
    return path


def snapshot(bids, asks, timestamp=0) -> BookSnapshot:
    update = OrderUpdate(timestamp, Side.BID, Action.ADD, "1", 15, 3)
    return BookSnapshot(update, bids, asks)


class TestReadingTheFeed:
    def test_rows_are_parsed_into_typed_records(self, tmp_path):
        path = write_csv(tmp_path / "day.csv", HEADER + "0,b,a,0,9990,11\n5,a,d,2,9995,1\n")

        assert list(read_updates(path)) == [
            OrderUpdate(0, Side.BID, Action.ADD, "0", 9990, 11),
            OrderUpdate(5, Side.ASK, Action.DELETE, "2", 9995, 1),
        ]

    def test_order_id_stays_text(self, tmp_path):
        """Ids are only ever dictionary keys, so they are never parsed as numbers."""
        path = write_csv(tmp_path / "day.csv", HEADER + "0,b,a,007,9990,11\n")

        assert next(read_updates(path)).order_id == "007"

    def test_unexpected_header_is_rejected(self, tmp_path):
        path = write_csv(tmp_path / "day.csv", "time,side,action,id,price,qty\n0,b,a,0,1,1\n")

        with pytest.raises(ValueError, match="unexpected header"):
            list(read_updates(path))

    def test_empty_file_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            list(read_updates(write_csv(tmp_path / "day.csv", "")))

    @pytest.mark.parametrize("row", ["1,b,a,1,not_a_price,5", "1,x,a,1,9990,5"])
    def test_malformed_row_reports_its_line_number(self, tmp_path, row):
        path = write_csv(tmp_path / "day.csv", HEADER + "0,b,a,0,9990,11\n" + row + "\n")

        with pytest.raises(ValueError, match="day.csv:3"):
            list(read_updates(path))


class TestWritingSnapshots:
    def test_columns_follow_the_specified_naming(self):
        columns = output_columns(5)

        assert columns[:3] == ["timestamp", "price", "side"]
        assert columns[3:7] == ["bp0", "bq0", "bp1", "bq1"]
        assert columns[13:17] == ["ap0", "aq0", "ap1", "aq1"]
        assert len(columns) == 23

    def test_missing_levels_are_padded_with_zero_quantity(self):
        row = snapshot_row(snapshot([Level(15, 3)], []), depth=5)

        assert row[:5] == [0, 15, "b", 15, 3]
        assert row[5:] == [EMPTY_LEVEL_PRICE, 0] * 9
        assert len(row) == len(output_columns(5))


class TestEndToEnd:
    def run(self, tmp_path) -> list[dict[str, str]]:
        source = write_csv(tmp_path / "in.csv", WORKED_EXAMPLE)
        replay_file(source, tmp_path / "out.csv")
        with (tmp_path / "out.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_one_output_row_per_input_row_echoing_the_update(self, tmp_path):
        rows = self.run(tmp_path)

        assert len(rows) == 5
        assert [row["timestamp"] for row in rows] == ["1", "2", "3", "4", "5"]
        assert [row["side"] for row in rows] == ["b", "b", "b", "a", "b"]
        assert [row["price"] for row in rows] == ["15", "15", "20", "30", "15"]

    def test_book_state_matches_the_documented_example(self, tmp_path):
        rows = self.run(tmp_path)

        assert (rows[0]["bp0"], rows[0]["bq0"]) == ("15", "3")
        assert (rows[1]["bp0"], rows[1]["bq0"]) == ("15", "8")
        assert (rows[2]["bp1"], rows[2]["bq1"]) == ("15", "5")
        assert (rows[3]["ap0"], rows[3]["aq0"]) == ("30", "1")
        assert (rows[4]["bp0"], rows[4]["bq0"]) == ("20", "3")
        assert rows[4]["bq1"] == "0"

    def test_absent_levels_report_zero_quantity(self, tmp_path):
        first = self.run(tmp_path)[0]

        assert [first[f"bq{index}"] for index in range(1, 5)] == ["0"] * 4
        assert [first[f"aq{index}"] for index in range(5)] == ["0"] * 5

    def test_each_run_starts_from_an_empty_book(self, tmp_path):
        assert self.run(tmp_path) == self.run(tmp_path)
