# AI usage

The assessment permits AI use and asks for the prompt history and an explanation
of how and why it was prompted. This is that record.

> **Note to the submitter:** this is a factual account of the sessions as they
> happened. Please read it and correct anything that misrepresents your own
> involvement before submitting.

## Summary

The code here was written with Claude (Claude Code), working from the
specification document and the supplied data across four sessions. The
direction, the decisions at each branch point, and the review were mine.

The working pattern was deliberately not "write me an order book":

1. establish what the data actually contains, from the data, not from assumption;
2. settle the design decisions explicitly before any code was written;
3. build in small pieces, each with tests;
4. trace every claim in the documentation back to a number the code prints.

## Session 1 - understanding the problem

Spent entirely on the specification and the semantics of the feed, with no code
written. The conclusions carried into the build were:

- `modify` gives an order's new **full state**, not a delta;
- order ids are unique per `(day, side)`, **not globally**;
- the aggregate book needs no per-order time priority;
- a separate live-order map is required, because modify and delete never carry
  the order's *previous* price.

That last point is the one everything else in Part 1 hangs off.

## Session 2 - building it

### The directing prompt

> "Now i need you to read through the requirements QTCodeTest.docx... You can
> make suggestions to change it if you think it is better, but in generally I
> just need a well done code base with good structure and readability for it to
> be scalable. I will be reading this codebase afterwards so make sure it is done
> with good software engineering practices. Ask me any questions if you are
> unsure before proceeding."

Phrased that way on purpose. Three parts of it did most of the work:

- **"read through the requirements"** - the specification is the source of truth,
  not a previous session's summary of it. The `.docx` was parsed directly rather
  than described.
- **"you can make suggestions to change it"** - the earlier design was explicitly
  made non-binding, so a better structure would not be argued out of existence by
  an earlier decision.
- **"ask me any questions before proceeding"** - the expensive failure mode on a
  task this size is a confident wrong turn taken silently in the first five
  minutes. Asking first is cheap.

### What happened before any code

An audit script was run over all five days - 1.18 million rows - to establish
what the feed actually does. It found: tick size 5, a one-tick spread on 76-96%
of updates, no executions anywhere in the feed, modifies that never move price,
up to 20 updates sharing a single microsecond, and a book that is never crossed.

Several of those findings redirected the work. The absence of trade data ruled
out a whole family of prediction targets. The one-tick spread is why the feature
set is built around queue imbalance. The repeated timestamps are why every time
lookup resolves ties to the *last* observation at an instant.

### The decisions I was asked to make

| Question | Chosen | Why |
| --- | --- | --- |
| Python or C++ for Part 1 | Python only | Keeps both parts one library, which is what the brief says it is grading. The finished build does 1.18M rows in about 8 seconds, so C++ would have bought nothing |
| How deep to take the model | Ridge, done properly | The brief asks for a simple model. Interpretable coefficients and honest validation were judged worth more than a better score from a heavier one |
| Package layout | One package, two subpackages | Shared types in one place, one import root, a visible Part 1 / Part 2 boundary |

### Where the AI was corrected

A clean account of AI use should include the parts that did not go straight
through:

- A test asserted that `imbalance_touch` would be the largest coefficient on
  synthetic data. It failed, correctly: the fixture made several features
  perfectly collinear, so which ranked first was arbitrary. Rewritten to assert
  what actually matters - that the fitted signal *follows* imbalance.
- The first sampler thinned the data by bucketing on `timestamp // spacing`. That
  guarantees one row per bucket but **not** a minimum gap between the rows chosen,
  which is the entire point. Replaced with a greedy forward scan, plus a test
  asserting the gap.
- The horizon was initially set to 10 seconds by assumption. The horizon scan was
  then built to decide it from data, and it moved the answer to 5 seconds.

### Changes made to the session 1 design

- **Bid prices are stored as true values, not negated.** The original sketch
  negated bid prices so both sides would sort best-first; storing real prices and
  branching once when slicing the top is easier to read and to debug.
- **Padding moved out of the book** into the CSV writer - it is a property of the
  output format, not of the book.
- **Updates are parsed into a typed record** at the boundary, rather than five
  positional arguments of raw strings reaching the book.
- **Integrity failures raise rather than pass silently.** A delete for an unknown
  order, an add reusing a live id, or a level driven negative each raise with
  context. Absorbing any of them leaves every later row subtly wrong with no way
  to tell from the output.

## Session 3 - cutting it back

Session 2 produced a working but sprawling submission: thirty source files, about
2,900 lines. The concern was not correctness but that I could not stand behind
every file in an interview, so I asked for options before deciding anything.

Measuring first was worth doing: only about half the lines were executable code,
the rest docstrings and blank lines. Cutting *lines* and cutting *complexity*
were different problems. Four cuts followed:

| Cut | Effect |
| --- | --- |
| Merge modules | 30 files to 18 |
| Trim docstrings | Module-length essays reduced to a few lines, with the reasoning living in the report rather than duplicated in both |
| Drop weak features | 22 features to 11 |
| Consolidate tests | Ten test modules to five; the five-step worked example became one readable walkthrough |

**The feature cut is the part worth defending.** Nine features were removed -
depth-weighted imbalance, log depth ratio, and rolling volatility, event rate and
cancellation ratio at each of three windows. The evidence was already in the
walk-forward output: every one had a coefficient near zero and five had signs
that *flipped between folds*. Removing them left out-of-sample performance
slightly **better**, which is what you expect when what you removed was fitting
noise. It also removed the only consumer of two extra Part 1 output columns, so
those went too - the reconstruction now emits exactly the columns the
specification asks for.

## Session 4 - the report, and a last cut

Writing the report surfaced one more problem. A correlation matrix of the
predictors showed `imbalance_touch` and `microprice_offset_ticks` correlating at
**0.99**, with variance inflation factors above 50 - unsurprising in hindsight,
since the micro-price offset is algebraically `(spread / 2) × imbalance` and the
spread is one tick for most of the session.

The fit had been behaving exactly as collinearity predicts: +0.12 on imbalance
and −0.06 on the micro-price, two large opposing coefficients that partly
cancelled and could not be read individually. Both alternatives were tested
rather than argued about:

| | IC | R² | Sign accuracy |
| --- | --- | --- | --- |
| Keep both (11 features) | 0.2615 | 0.0668 | 77.2% |
| **Drop micro-price (10)** | **0.2603** | **0.0662** | **77.4%** |
| Drop imbalance (10) | 0.2565 | 0.0644 | 77.6% |

Dropping the micro-price cost 0.001 of IC. Imbalance's coefficient settled to a
single readable +0.063 and the regression's condition number fell from 18.7 to
5.15.

The codebase was then stripped to what the report actually refers to: the
per-row invariant checker and its `--validate` flag were removed, along with
several unused accessors and helpers, and the separate modelling write-up was
folded into `report.html`.

## What was not delegated

Every number in `README.md` and `report.html` is the printed output of the two
commands documented in the README, not a recalled or estimated figure. Where a
claim is made about a trade-off - the horizon, the feature cut, the subsampling -
it was tested and the losing option is recorded above.
