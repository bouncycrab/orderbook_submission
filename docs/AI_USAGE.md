# AI usage

The assessment permits AI use and asks for the prompt history and an explanation
of how and why it was prompted. This is that record.


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

> " Read through the requirements stated at [QuestionFile] You can
> make suggestions to change it if you think it is better, but in generally I
>  need a well done code base with good structure and readability for it to
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

- The horizon was initially set to 10 seconds by default. The horizon scan was
  then built to decide it from data, and it moved the answer to 5 seconds. (See the report.html for the scan.)
- AI used produced correlated features. I verified using a correlation matrix and variance inflation factors were computed, and the micro-price feature was dropped to produce a more readable model.

Every number in `README.md` and `report.html` is the printed output of the two
commands documented in the README, not a recalled or estimated figure. 
