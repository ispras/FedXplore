from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from ui.analytics import (
    MetricPoint,
    load_final_metrics,
    metric_points_frame,
    normalize_metric_history,
)
from ui.comparison import (
    build_config_diff,
    experiment_config_from_spec,
    flatten_config,
)


class AnalyticsTests(unittest.TestCase):
    def test_normalize_metric_history_coerces_valid_entities_and_skips_invalid_values(
        self,
    ) -> None:
        history = [
            {
                "key": "train.loss",
                "value": "0.75",
                "step": "2",
                "timestamp": "1700000000123",
            },
            SimpleNamespace(value=0.5, step=3, timestamp=1700000000456),
            {"value": None, "step": 4, "timestamp": 1700000000789},
            {"value": "not-a-number", "step": 5, "timestamp": 1700000000999},
            {"value": math.inf, "step": 6, "timestamp": 1700000001111},
            {"value": math.nan, "step": 7, "timestamp": 1700000001222},
        ]

        points = normalize_metric_history("fallback.metric", history, run_id="run-1")

        self.assertEqual(
            points,
            [
                MetricPoint(
                    metric="train.loss",
                    value=0.75,
                    step=2.0,
                    timestamp=1700000000123,
                    run_id="run-1",
                ),
                MetricPoint(
                    metric="fallback.metric",
                    value=0.5,
                    step=3.0,
                    timestamp=1700000000456,
                    run_id="run-1",
                ),
            ],
        )

    def test_metric_points_frame_uses_step_per_metric_and_timestamp_fallback(
        self,
    ) -> None:
        points = [
            MetricPoint("loss", 0.4, 2, 2_000, "run-a"),
            MetricPoint("loss", 0.6, 1, 1_000, "run-a"),
            MetricPoint("loss", 0.5, None, 1_500, "run-b"),
            MetricPoint("accuracy", 0.7, 2, 2_000, "run-a"),
            MetricPoint("accuracy", 0.8, 1, 1_000, "run-a"),
            MetricPoint("accuracy", 0.75, 1, 1_500, "run-b"),
        ]

        frame = metric_points_frame(
            points,
            run_labels={"run-a": "Run A", "run-b": "Run B"},
        )

        loss = frame.loc[frame["metric"] == "loss"]
        accuracy = frame.loc[frame["metric"] == "accuracy"]
        self.assertEqual(set(loss["x_axis"]), {"Timestamp"})
        self.assertEqual(
            [value.isoformat() for value in loss.loc[loss["run_id"] == "run-a", "x"]],
            [
                "1970-01-01T00:00:01+00:00",
                "1970-01-01T00:00:02+00:00",
            ],
        )
        self.assertEqual(set(accuracy["x_axis"]), {"Step"})
        self.assertEqual(
            list(accuracy.loc[accuracy["run_id"] == "run-a", "x"]),
            [1.0, 2.0],
        )
        self.assertEqual(set(frame["run_label"]), {"Run A", "Run B"})

    def test_load_final_metrics_prefers_latest_step_then_timestamp_fallback(self) -> None:
        points = [
            MetricPoint("accuracy", 0.70, 1, 1_000),
            MetricPoint("accuracy", 0.80, 2, 1_500),
            MetricPoint("accuracy", 0.82, 2, 2_000),
            MetricPoint("loss", 0.70, None, 1_000),
            MetricPoint("loss", 0.50, None, 2_000),
            # A series with a missing step is unusable as a whole and falls
            # back to timestamp rather than mixing a round axis and time.
            MetricPoint("mixed", 0.20, 5, 3_000),
            MetricPoint("mixed", 0.10, None, 4_000),
        ]

        final_by_metric = {
            item.metric: item for item in load_final_metrics(points)
        }

        self.assertEqual(final_by_metric["accuracy"].value, 0.82)
        self.assertEqual(final_by_metric["accuracy"].step, 2)
        self.assertEqual(final_by_metric["accuracy"].timestamp, 2_000)
        self.assertEqual(final_by_metric["loss"].value, 0.50)
        self.assertIsNone(final_by_metric["loss"].step)
        self.assertEqual(final_by_metric["loss"].timestamp, 2_000)
        self.assertEqual(final_by_metric["mixed"].value, 0.10)
        self.assertIsNone(final_by_metric["mixed"].step)


class ComparisonTests(unittest.TestCase):
    def test_flatten_config_uses_dotted_paths_and_stable_complex_cells(self) -> None:
        flattened = flatten_config(
            {
                "dataset": {
                    "name": "cifar10",
                    "alpha": 0.1,
                    "classes": [3, 1],
                },
                "selector": {"enabled": True, "options": {"seed": None}},
                "attack": {"label_map": {"b": 2, "a": 1}},
            }
        )

        self.assertEqual(
            flattened,
            {
                "dataset.name": "cifar10",
                "dataset.alpha": "0.1",
                "dataset.classes": "[3, 1]",
                "selector.enabled": "true",
                "selector.options.seed": "null",
                "attack.label_map.a": "1",
                "attack.label_map.b": "2",
            },
        )

    def test_build_config_diff_compares_any_number_of_runs_and_marks_missing_values(
        self,
    ) -> None:
        rows = build_config_diff(
            {
                "run-a": {
                    "dataset": {"name": "cifar10", "alpha": 0.1},
                    "selector": {"name": "uniform"},
                },
                "run-b": {
                    "dataset": {"name": "cifar10", "alpha": 0.2},
                    "selector": {"name": "pow"},
                },
                "run-c": {
                    "dataset": {"name": "cifar10"},
                    "selector": {"name": "uniform"},
                },
            }
        )
        rows_by_parameter = {row.parameter: row for row in rows}

        self.assertEqual(
            [row.parameter for row in rows],
            ["dataset.alpha", "dataset.name", "selector.name"],
        )
        self.assertTrue(rows_by_parameter["dataset.alpha"].differs)
        self.assertEqual(
            rows_by_parameter["dataset.alpha"].values,
            {"run-a": "0.1", "run-b": "0.2", "run-c": "N/A"},
        )
        self.assertFalse(rows_by_parameter["dataset.name"].differs)
        self.assertTrue(rows_by_parameter["selector.name"].differs)

    def test_experiment_config_prefers_saved_overrides_and_resolves_duplicates(
        self,
    ) -> None:
        config = experiment_config_from_spec(
            {
                "overrides": [
                    "+model=resnet18",
                    "optimizer=adam",
                    "model=resnet50",
                    "not-an-assignment",
                ],
                "form_payload": {"model": "ignored-form-value", "seed": 42},
                "run_id": "run-1",
            }
        )

        self.assertEqual(
            config,
            {"model": "resnet50", "optimizer": "adam"},
        )

    def test_experiment_config_falls_back_to_form_then_sanitized_legacy_spec(self) -> None:
        self.assertEqual(
            experiment_config_from_spec(
                {
                    "overrides": ["not-an-assignment"],
                    "form_payload": {"dataset": "cifar10", "seed": 42},
                    "run_id": "run-1",
                }
            ),
            {"dataset": "cifar10", "seed": 42},
        )
        self.assertEqual(
            experiment_config_from_spec(
                {
                    "dataset": "cifar10",
                    "seed": 42,
                    "run_id": "run-1",
                    "created_at": "2026-09-14T12:00:00+00:00",
                    "mlflow_url": "http://mlflow.example.invalid",
                }
            ),
            {"dataset": "cifar10", "seed": 42},
        )


if __name__ == "__main__":
    unittest.main()
