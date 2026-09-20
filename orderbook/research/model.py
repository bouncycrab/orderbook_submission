"""The model, and how it is measured.

**The model** is standardised features into a ridge regression, deliberately
nothing more. The features are collinear by construction - imbalance at the
touch and the micro-price offset describe overlapping things - and under
collinearity an unpenalised fit produces large coefficients of opposite sign
that cancel, unstable between days and impossible to read. The penalty shrinks
them towards each other: a little in-sample fit for coefficients that can be
interpreted and that survive an unseen day. Standardising first makes them read
as "ticks of predicted move per standard deviation of feature".

**Validation** is walk-forward by trading day: fit on days up to a point, test on
the next. Random k-fold would let the model learn from the afternoon to predict
the morning, and with features built from overlapping trailing windows it would
leak neighbouring rows across fold boundaries. Splitting whole days avoids both.

Scores are expected to be small. A short-horizon book signal explaining a few
per cent of forward variance is doing well; a large R-squared here would be
evidence of leakage, not insight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ResearchConfig
from .dataset import Dataset, DayPanel, SamplingPolicy, assemble_dataset


class SignalModel:
    """Standardiser and ridge regression, fitted as one pipeline.

    The scaler sits inside the pipeline so it only ever sees training data.
    """

    def __init__(self, alphas: tuple[float, ...]) -> None:
        self._pipeline = Pipeline(
            [("scale", StandardScaler()), ("ridge", RidgeCV(alphas=alphas))]
        )
        self._feature_names: list[str] = []

    @classmethod
    def from_config(cls, config: ResearchConfig) -> "SignalModel":
        return cls(alphas=config.ridge_alphas)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SignalModel":
        self._feature_names = list(features.columns)
        self._pipeline.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predicted forward mid-price change, in ticks."""
        return self._pipeline.predict(features)

    @property
    def alpha(self) -> float:
        """Penalty chosen by cross-validation within the training data."""
        return float(self._pipeline.named_steps["ridge"].alpha_)

    @property
    def intercept(self) -> float:
        return float(self._pipeline.named_steps["ridge"].intercept_)

    @property
    def coefficients(self) -> pd.Series:
        """Standardised coefficients, largest magnitude first."""
        ridge = self._pipeline.named_steps["ridge"]
        coefficients = pd.Series(ridge.coef_, index=self._feature_names)
        return coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)


def ols_summary(dataset: Dataset):
    """Unpenalised OLS on standardised features, with inference.

    The signal itself is fitted with ridge, which shrinks coefficients and so
    has no standard errors worth quoting. This is the companion diagnostic: the
    same design matrix fitted without a penalty, so each coefficient comes with
    a standard error, a t-statistic and a 95% confidence interval.

    Those intervals mean something here only because the sample was thinned to
    non-overlapping forward windows. Fitted on every row, the residuals would be
    heavily autocorrelated and the standard errors badly understated - the
    coefficients would look far more certain than they are.

    Features are standardised first, so coefficients read in ticks per standard
    deviation and are comparable with each other.
    """
    import statsmodels.api as sm

    standardised = (dataset.features - dataset.features.mean()) / dataset.features.std()
    design = sm.add_constant(standardised)
    return sm.OLS(dataset.target.to_numpy(), design).fit()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() == 0 or right.std() == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    # Guarded rather than left to scipy, which warns on constant input; a
    # constant prediction is a legitimate degenerate outcome, not a problem.
    if left.std() == 0 or right.std() == 0:
        return np.nan
    return float(stats.spearmanr(left, right).statistic)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Summarise prediction quality.

    ``r2_vs_zero`` sits alongside the conventional R-squared because the honest
    baseline for a trading signal is predicting no move at all, not the mean of a
    sample nobody knew in advance. Where they diverge, believe the first.

    ``mean_signed_return_ticks`` is what the signal captures per observation if
    traded at its own sign, before costs - to be read against the half-spread.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    residual = float(np.sum((actual - predicted) ** 2))
    about_mean = float(np.sum((actual - actual.mean()) ** 2))
    about_zero = float(np.sum(actual**2))

    moved = actual != 0
    directional = np.sign(predicted[moved]) == np.sign(actual[moved])

    return {
        "rows": float(actual.size),
        "r2": 1.0 - residual / about_mean if about_mean else np.nan,
        "r2_vs_zero": 1.0 - residual / about_zero if about_zero else np.nan,
        "pearson_ic": _correlation(predicted, actual),
        "spearman_ic": _rank_correlation(predicted, actual),
        "sign_accuracy": float(directional.mean()) if moved.any() else np.nan,
        "mean_abs_prediction_ticks": float(np.abs(predicted).mean()),
        "mean_signed_return_ticks": float(np.mean(np.sign(predicted) * actual)),
    }


def bucket_profile(actual: np.ndarray, predicted: np.ndarray, buckets: int) -> pd.DataFrame:
    """Mean realised move within each bucket of predicted move.

    A monotone profile is the practical test of a signal: the ranking is
    informative even where the fitted magnitudes are not.
    """
    frame = pd.DataFrame({"predicted": predicted, "actual": actual})
    ranks = frame["predicted"].rank(method="first")
    frame["bucket"] = pd.qcut(ranks, buckets, labels=False, duplicates="drop")

    return (
        frame.groupby("bucket")
        .agg(
            rows=("actual", "size"),
            mean_predicted_ticks=("predicted", "mean"),
            mean_actual_ticks=("actual", "mean"),
        )
        .reset_index()
    )


@dataclass
class FoldResult:
    """Outcome of fitting on earlier days and testing on one later day."""

    test_day: str
    train_days: list[str]
    alpha: float
    metrics: dict[str, float]
    coefficients: pd.Series
    predictions: np.ndarray = field(repr=False)
    actuals: np.ndarray = field(repr=False)
    timestamps: np.ndarray = field(repr=False, default=None)


@dataclass
class WalkForwardResult:
    folds: list[FoldResult]

    @property
    def metrics_frame(self) -> pd.DataFrame:
        rows = [
            {"test_day": fold.test_day, "alpha": fold.alpha, **fold.metrics}
            for fold in self.folds
        ]
        return pd.DataFrame(rows).set_index("test_day")

    @property
    def pooled_metrics(self) -> dict[str, float]:
        return regression_metrics(
            np.concatenate([fold.actuals for fold in self.folds]),
            np.concatenate([fold.predictions for fold in self.folds]),
        )

    @property
    def mean_coefficients(self) -> pd.Series:
        """Average coefficient across folds.

        Stability matters more than size here: a feature whose sign flips
        between folds is not telling you anything tradeable.
        """
        coefficients = pd.concat([fold.coefficients for fold in self.folds], axis=1)
        means = coefficients.mean(axis=1)
        return means.reindex(means.abs().sort_values(ascending=False).index)

    @property
    def predictions_frame(self) -> pd.DataFrame:
        """Every out-of-sample prediction with the row it belongs to.

        The first trading day never appears: it is only ever training data, so
        no honest prediction exists for it.
        """
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "day": fold.test_day,
                        "timestamp": fold.timestamps,
                        "predicted_ticks": fold.predictions,
                        "actual_ticks": fold.actuals,
                    }
                )
                for fold in self.folds
            ],
            ignore_index=True,
        )

    def calibration(self, bins: int = 30) -> pd.DataFrame:
        """Mean actual against mean predicted, in equal-count bins.

        The readable form of a predicted-versus-actual scatter when the target is
        discrete and mostly zero: a raw scatter collapses into horizontal stripes
        at the few values the mid can take, which hides the relationship instead
        of showing it. Binning by prediction and averaging recovers it.
        """
        frame = self.predictions_frame
        ranks = frame["predicted_ticks"].rank(method="first")
        frame = frame.assign(bin=pd.qcut(ranks, bins, labels=False, duplicates="drop"))
        return (
            frame.groupby("bin")
            .agg(
                rows=("actual_ticks", "size"),
                predicted=("predicted_ticks", "mean"),
                actual=("actual_ticks", "mean"),
                actual_se=("actual_ticks", lambda column: column.std() / np.sqrt(column.size)),
            )
            .reset_index(drop=True)
        )

    def pooled_bucket_profile(self, buckets: int) -> pd.DataFrame:
        return bucket_profile(
            np.concatenate([fold.actuals for fold in self.folds]),
            np.concatenate([fold.predictions for fold in self.folds]),
            buckets,
        )


def walk_forward(dataset: Dataset, config: ResearchConfig) -> WalkForwardResult:
    """Fit on an expanding window of days, testing on the next day each time."""
    days = dataset.days
    if len(days) < 2:
        raise ValueError("walk-forward validation needs at least two trading days")

    folds = []
    for split in range(1, len(days)):
        train = dataset.rows_for_days(days[:split])
        test = dataset.rows_for_days([days[split]])

        model = SignalModel.from_config(config).fit(train.features, train.target)
        predictions = model.predict(test.features)
        actuals = test.target.to_numpy()

        folds.append(
            FoldResult(
                test_day=days[split],
                train_days=list(days[:split]),
                alpha=model.alpha,
                metrics=regression_metrics(actuals, predictions),
                coefficients=model.coefficients,
                predictions=predictions,
                actuals=actuals,
                timestamps=test.timestamp.to_numpy(),
            )
        )

    return WalkForwardResult(folds)


def scan_horizons(
    panels: list[DayPanel],
    config: ResearchConfig,
    policy: SamplingPolicy = "time_grid",
) -> pd.DataFrame:
    """Re-run walk-forward validation at each candidate horizon.

    This is what the choice of horizon rests on: identical features and
    validation, only the target's horizon changing. Prepared panels are reused,
    so features are built once rather than once per horizon.
    """
    rows = []
    for horizon in config.candidate_horizons_seconds:
        horizon_config = config.with_horizon(horizon)
        dataset = assemble_dataset(panels, horizon_config, policy)
        if len(dataset.days) < 2:
            continue

        pooled = walk_forward(dataset, horizon_config).pooled_metrics
        rows.append(
            {
                "horizon_seconds": horizon,
                "rows": len(dataset),
                "target_std_ticks": float(dataset.target.std()),
                **{
                    key: pooled[key]
                    for key in (
                        "r2_vs_zero",
                        "pearson_ic",
                        "sign_accuracy",
                        "mean_signed_return_ticks",
                    )
                },
            }
        )

    return pd.DataFrame(rows).set_index("horizon_seconds")
