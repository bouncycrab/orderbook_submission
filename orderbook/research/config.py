"""Research configuration - every modelling choice in one place.

One frozen dataclass rather than constants scattered through the modules, so an
experiment is fully described by a single object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

MICROSECONDS_PER_SECOND = 1_000_000


@dataclass(frozen=True)
class ResearchConfig:
    """Parameters defining a single signal-fitting experiment."""

    #: Minimum price increment, confirmed as 5 by taking the GCD of all price
    #: differences on each day. Targets and price features are in ticks so they
    #: are comparable across days.
    tick_size: int = 5

    #: Price levels per side in the Part 1 output.
    depth: int = 5

    #: Prediction horizon, in wall-clock seconds rather than a number of events:
    #: event intensity varies several-fold through the day, so "20 updates
    #: ahead" means different amounts of real time at the open and at noon.
    #:
    #: Five seconds is where the horizon scan peaks - out-of-sample IC and
    #: R-squared are both highest there, on a sample twice the size of the
    #: ten-second one. Below it the target is dominated by discreteness, above it
    #: book state is diluted by unrelated drift. See ``--scan-horizons``.
    horizon_seconds: float = 5.0

    #: Horizons compared before settling on the above, reported by the scan so
    #: the choice can be seen rather than asserted.
    candidate_horizons_seconds: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0)

    #: Trailing windows for the dynamic features, spread over two orders of
    #: magnitude to catch both fast order flow and slower drift.
    lookback_windows_seconds: tuple[float, ...] = (1.0, 5.0, 30.0)

    #: Spacing of the fitting sample on the clock; defaults to one horizon, so
    #: consecutive rows have non-overlapping forward windows.
    sample_spacing_seconds: float | None = None

    #: Ignored period at the start of each day, while the book is still filling
    #: in from empty and its depth is not comparable with the rest of the day.
    warmup_seconds: float = 60.0

    #: Ridge penalties searched over.
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)

    #: Buckets used when reporting realised move by predicted decile.
    profile_buckets: int = 10

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")

    @property
    def horizon_us(self) -> int:
        return int(self.horizon_seconds * MICROSECONDS_PER_SECOND)

    @property
    def warmup_us(self) -> int:
        return int(self.warmup_seconds * MICROSECONDS_PER_SECOND)

    @property
    def spacing_us(self) -> int:
        spacing = (
            self.sample_spacing_seconds
            if self.sample_spacing_seconds is not None
            else self.horizon_seconds
        )
        return int(spacing * MICROSECONDS_PER_SECOND)

    def window_us(self, seconds: float) -> int:
        return int(seconds * MICROSECONDS_PER_SECOND)

    def with_horizon(self, horizon_seconds: float) -> "ResearchConfig":
        """Copy at a different horizon, for the horizon scan."""
        return replace(self, horizon_seconds=horizon_seconds)
