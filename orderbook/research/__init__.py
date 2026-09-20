"""Part 2: features, targets and a predictive signal built on reconstructed books.

The pipeline is linear and each stage is usable on its own:

    load  ->  features  ->  target  ->  sample  ->  fit  ->  evaluate

:mod:`config` holds every modelling choice, :mod:`features` the causal windowing
primitives and the features themselves, :mod:`dataset` the loading, subsampling
and assembly, and :mod:`model` the fit and its evaluation.
"""

from .config import ResearchConfig
from .dataset import (
    Dataset,
    DayPanel,
    assemble_dataset,
    build_panels,
    load_book_day,
)
from .features import build_features
from .model import (
    SignalModel,
    WalkForwardResult,
    bucket_profile,
    ols_summary,
    regression_metrics,
    scan_horizons,
    walk_forward,
)
from .targets import forward_mid_change

__all__ = [
    "Dataset",
    "DayPanel",
    "ResearchConfig",
    "SignalModel",
    "WalkForwardResult",
    "assemble_dataset",
    "bucket_profile",
    "build_features",
    "build_panels",
    "forward_mid_change",
    "load_book_day",
    "ols_summary",
    "regression_metrics",
    "scan_horizons",
    "walk_forward",
]
