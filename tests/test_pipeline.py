"""End-to-end: assembly, fitting and evaluation on synthetic data."""

import numpy as np
import pytest
from conftest import TICK, make_book_frame

from orderbook.research.config import ResearchConfig
from orderbook.research.dataset import assemble_dataset, build_panel
from orderbook.research.model import (
    SignalModel,
    bucket_profile,
    regression_metrics,
    walk_forward,
)

STEP_US = 100_000
BLOCK = 40


def trending_day(day: str, blocks: int = 40, seed: int = 0):
    """A book whose mid trends in the direction its touch imbalance points.

    The direction persists for blocks of forty updates, so over a one second
    horizon the forward move is essentially determined by the imbalance visible
    now. A pipeline with a sign error or an off-by-one in its windows will not
    find that; a correct one must.
    """
    rng = np.random.default_rng(seed)
    directions = np.repeat(rng.choice([-1, 1], size=blocks), BLOCK)
    base = 10_000 + TICK * np.cumsum(np.concatenate(([0], directions[:-1])))

    return make_book_frame(
        timestamps_us=[index * STEP_US for index in range(directions.size)],
        bid_prices=list(base),
        bid_sizes=list(np.where(directions > 0, 30, 10)),
        ask_prices=list(base + TICK),
        ask_sizes=list(np.where(directions > 0, 10, 30)),
        day=day,
    )


@pytest.fixture
def pipeline_config() -> ResearchConfig:
    return ResearchConfig(
        tick_size=TICK,
        depth=5,
        horizon_seconds=1.0,
        lookback_windows_seconds=(1.0,),
        warmup_seconds=0.0,
    )


@pytest.fixture
def panels(pipeline_config):
    return [
        build_panel(trending_day("20190610", seed=1), pipeline_config),
        build_panel(trending_day("20190611", seed=2), pipeline_config),
    ]


class TestDatasetAssembly:
    def test_sample_is_thinned_and_every_row_is_usable(self, panels, pipeline_config):
        dataset = assemble_dataset(panels, pipeline_config)

        assert 0 < len(dataset) < sum(len(panel) for panel in panels)
        assert np.isfinite(dataset.target.to_numpy()).all()
        assert np.isfinite(dataset.features.to_numpy()).all()

    def test_days_are_labelled_and_selectable(self, panels, pipeline_config):
        dataset = assemble_dataset(panels, pipeline_config)

        assert dataset.days == ["20190610", "20190611"]
        assert dataset.rows_for_days(["20190610"]).days == ["20190610"]

    def test_rows_respect_the_horizon_spacing(self, panels, pipeline_config):
        dataset = assemble_dataset(panels, pipeline_config)

        for day in dataset.days:
            stamps = dataset.rows_for_days([day]).timestamp.to_numpy()
            assert (np.diff(stamps) >= pipeline_config.spacing_us).all()


class TestModel:
    def test_model_recovers_a_known_relationship(self, panels, pipeline_config):
        dataset = assemble_dataset(panels, pipeline_config)

        model = SignalModel.from_config(pipeline_config).fit(dataset.features, dataset.target)

        assert np.corrcoef(model.predict(dataset.features), dataset.target)[0, 1] > 0.9

    def test_signal_follows_touch_imbalance(self, panels, pipeline_config):
        """Asserted through the prediction rather than by ranking coefficients:
        several features here encode the same direction exactly, and ridge
        spreads weight across collinear inputs, so which is largest is
        arbitrary. What the signal does is not."""
        dataset = assemble_dataset(panels, pipeline_config)

        model = SignalModel.from_config(pipeline_config).fit(dataset.features, dataset.target)
        predictions = model.predict(dataset.features)

        assert np.corrcoef(predictions, dataset.features["imbalance_touch"])[0, 1] > 0.9
        assert model.coefficients["imbalance_touch"] > 0
        assert set(model.coefficients.index) == set(dataset.features.columns)


class TestWalkForward:
    def test_fits_on_earlier_days_and_generalises_to_the_next(self, panels, pipeline_config):
        dataset = assemble_dataset(panels, pipeline_config)

        result = walk_forward(dataset, pipeline_config)

        assert [fold.test_day for fold in result.folds] == ["20190611"]
        assert result.folds[0].train_days == ["20190610"]
        assert result.pooled_metrics["pearson_ic"] > 0.5

    def test_a_single_day_cannot_be_validated(self, panels, pipeline_config):
        dataset = assemble_dataset(panels[:1], pipeline_config)

        with pytest.raises(ValueError, match="at least two trading days"):
            walk_forward(dataset, pipeline_config)


class TestMetrics:
    def test_perfect_prediction_scores_one(self):
        actual = np.array([-1.0, 0.5, 2.0, -0.5])

        metrics = regression_metrics(actual, actual)

        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["pearson_ic"] == pytest.approx(1.0)
        assert metrics["sign_accuracy"] == pytest.approx(1.0)

    def test_predicting_zero_throughout_scores_nothing(self):
        actual = np.array([-1.0, 0.5, 2.0, -0.5])

        metrics = regression_metrics(actual, np.zeros_like(actual))

        assert metrics["r2_vs_zero"] == pytest.approx(0.0)
        assert np.isnan(metrics["pearson_ic"])

    def test_sign_accuracy_ignores_rows_that_did_not_move(self):
        metrics = regression_metrics(np.array([0.0, 0.0, 1.0]), np.array([-5.0, 5.0, 1.0]))

        assert metrics["sign_accuracy"] == pytest.approx(1.0)

    def test_bucket_profile_is_monotone_for_a_perfect_signal(self):
        actual = np.linspace(-1, 1, 200)

        profile = bucket_profile(actual, actual, buckets=5)

        assert profile["mean_actual_ticks"].is_monotonic_increasing
        assert len(profile) == 5
