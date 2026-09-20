# Order Book Reconstruction and Signal Research

Submission for the two-part assessment:

- **Part 1** - reconstruct the aggregate limit order book from a day's raw order
  updates, emitting the best five price levels per side for every update.
- **Part 2** - build predictive features on that output, fit a signal, and
  measure it out of sample.

Both parts are one library. `orderbook/` holds the book engine, and
`orderbook/research/` the modelling work built on top of it.

## Start here

| If you want to... | Open |
| --- | --- |
| Read the reasoning and results | **`report.html`** - open in any browser |
| Run the code | the two commands below |
| See how AI was used | `docs/AI_USAGE.md` |
| Judge the code | `orderbook/book.py`, then `orderbook/research/features.py` |

`report.html` is the write-up: what the target is and why, the horizon scan
behind the five-second window, all ten features with their arithmetic worked
through on a real row, the collinearity evidence, and the fitted model with
confidence intervals. Every figure in it comes from the commands below.

## Setup

```bash
conda env create -f environment.yml
conda activate qt
```

or, into an existing environment:

```bash
pip install -r requirements.txt
```

Python 3.11 or newer is required (the book uses `StrEnum`). Dependencies are
numpy, pandas, scipy, scikit-learn, statsmodels, sortedcontainers and pytest.

## Running it

**Part 1** - one output file per input day, book starting empty each day:

```bash
python -m orderbook.cli.build_books res_*.csv --output-dir books
```

Writes `books/book_<date>.csv`, one row per input row, with the columns the
specification names. All five sample days rebuild in about eight seconds.

**Part 2** - features, fit, and the evaluation report:

```bash
python -m orderbook.cli.fit_signal books/book_*.csv --scan-horizons --summary
```

Prints the horizon scan, the fitting sample, walk-forward validation per day,
pooled out-of-sample metrics, coefficients, the calibration table and the decile
profile. `--summary` adds an OLS fit carrying standard errors and 95% confidence
intervals.

| Flag | Effect |
| --- | --- |
| `--horizon SECONDS` | prediction horizon (default 5) |
| `--sampling {time_grid,book_change,all}` | which rows to fit on (default `time_grid`) |
| `--scan-horizons` | compare candidate horizons before reporting the chosen one |
| `--summary` | OLS fit with standard errors and confidence intervals |
| `--export PATH` | write the fitting sample, features and target, to a CSV |
| `--depth N` | levels per side, Part 1 only (default 5) |

**Tests:**

```bash
pytest
```

73 cases, a fraction of a second, no data files needed.

## Layout

```
orderbook/
  book.py            domain types, PriceLadder, OrderBook
  io.py              parsing the feed, replay, writing output
  cli/
    build_books.py   Part 1 entry point
    fit_signal.py    Part 2 entry point
  research/
    config.py        every modelling choice, in one dataclass
    features.py      causal time-window primitives and the ten features
    targets.py       the prediction target
    dataset.py       loading, subsampling, assembling fitting samples
    model.py         ridge pipeline, walk-forward validation, metrics
tests/               six modules
docs/AI_USAGE.md     how AI was used, with prompts
report.html          the write-up
```

Twelve source files, plus six test modules.

## The two things worth knowing about the design

**Part 1 keeps a live-order map.** A `delete` or `modify` row gives the order's
*new* state but never says which price level it is currently resting on. Without
`OrderBook._live_orders` remembering that, a modify which moves an order between
prices cannot be unwound, and every subsequent row is silently wrong. That map,
not the price ladders, is the part of the design that matters.

**Part 2 fits on a thinned sample.** A five-second forward target is shared
almost in full by every row inside any five-second window, so fitting on all
1.18M rows would mean many overlapping views of far fewer independent events,
inflating every standard error and validation score. Rows are therefore spaced at
least one horizon apart. The Durbin-Watson statistic of 2.016 in the `--summary`
output is the confirmation that this worked.

Two tests check properties rather than values, and are the ones to read first:

- `tests/test_features.py::test_features_never_use_information_from_later_rows`
  truncates the data and asserts every earlier feature value is unchanged - a
  direct test that nothing looks ahead.
- `tests/test_dataset.py::TestTimeGrid::test_selected_rows_are_at_least_one_spacing_apart`
  asserts sampled rows are always at least one horizon apart.

`tests/test_book.py::test_specification_worked_example` is the five-step scenario
from the assessment document, asserted step by step.

## Results

Reconstruction matches the specification's worked example exactly, verified as a
unit test. The signal predicts the change in mid-price over the next five
seconds, in ticks, from ten features, validated walk-forward by day - fit on days
up to a point, test on the next, never the reverse.

| Metric (pooled out-of-sample, 22,361 rows) | Value |
| --- | --- |
| R² against a zero forecast | 0.066 |
| Pearson information coefficient | 0.260 |
| Spearman information coefficient | 0.267 |
| Sign accuracy, on rows where the mid moved | 77.4% |
| Mean captured move at the signal's own sign | 0.072 ticks |

Mean realised move rises monotonically across all ten deciles of predicted move,
and every coefficient is significant at the 5% level.

The last row is the caveat worth stating plainly: crossing a one-tick spread
costs about 0.5 ticks and the signal captures 0.072. This is a real predictive
relationship, not a trading strategy. `report.html` sets out what would have to
change for it to become one.
