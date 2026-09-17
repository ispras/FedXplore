from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.utils.logging_utils import (
    BaseLogger,
    MLFlowLogger,
    build_client_participation_histogram,
)


class MLFlowLoggingTests(unittest.TestCase):
    def test_initialization_writes_metadata_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)

            def initialize_base(logger, path):
                logger.run_dir = path
                logger.run_command = "python src/train.py"
                logger.checkpoint_path = None

            started_run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
            experiment = SimpleNamespace(experiment_id="experiment-456")
            with (
                patch.object(BaseLogger, "__init__", initialize_base),
                patch("src.utils.logging_utils.mlflow.set_tracking_uri"),
                patch(
                    "src.utils.logging_utils.mlflow.get_tracking_uri",
                    return_value=str(run_dir / "mlruns"),
                ),
                patch(
                    "src.utils.logging_utils.mlflow.set_experiment",
                    return_value=experiment,
                ),
                patch("src.utils.logging_utils.mlflow.active_run", return_value=None),
                patch(
                    "src.utils.logging_utils.mlflow.start_run",
                    return_value=started_run,
                ) as start_run,
            ):
                logger = MLFlowLogger(
                    run_dir=str(run_dir),
                    tracking_uri=str(run_dir / "mlruns"),
                    experiment_name="FedXplore Toy Examples",
                    run_name="group/condition",
                    tags={"seed": 42, "condition": "condition"},
                )

            start_run.assert_called_once_with(
                run_name="group/condition",
                tags={"seed": "42", "condition": "condition"},
            )
            metadata = json.loads(
                (run_dir / "mlflow_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata,
                {
                    "run_id": "run-123",
                    "experiment_id": "experiment-456",
                    "tracking_uri": str(run_dir / "mlruns"),
                    "run_url": "",
                    "run_name": "group/condition",
                },
            )
            self.assertEqual(logger.run_id, "run-123")

    def test_dataframe_metrics_are_logged_in_one_batch_with_canonical_names(self):
        logger = MLFlowLogger.__new__(MLFlowLogger)
        logger._pending_metrics = {}
        logger._pending_metric_step = None
        metrics = pd.DataFrame(
            {"overall": [0.75, 0.6]},
            index=["Accuracy", "f1-score"],
        )

        with patch("src.utils.logging_utils.mlflow.log_metrics") as log_metrics:
            logger.log_scalar(0.2, "val/loss", 3)
            logger.log_pandas(metrics, "val/", 3)

        log_metrics.assert_called_once_with(
            {
                "val/loss": 0.2,
                "val/Accuracy_overall": 0.75,
                "val/f1-score_overall": 0.6,
            },
            step=3,
        )


class ClientParticipationHistogramTests(unittest.TestCase):
    def test_clean_client_map_produces_single_color_and_clean_summary(self):
        selections = pd.DataFrame(
            {"round": [0, 1], "clients": [[0, 2], [0, 1]]}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "participation_histogram.png"
            with patch("src.utils.logging_utils.plt.bar") as bar:
                summary = build_client_participation_histogram(
                    selections,
                    num_clients=3,
                    save_path=output_path,
                    client_attack_map={0: "no_attack", 1: "no_attack", 2: "no_attack"},
                )

        self.assertEqual(bar.call_args.kwargs["color"], ["tab:blue"] * 3)
        pd.testing.assert_frame_equal(
            summary,
            pd.DataFrame(
                {
                    "client_id": [0, 1, 2],
                    "rounds_selected": [2, 1, 1],
                    "is_attacker": [False, False, False],
                    "attack_type": ["no_attack", "no_attack", "no_attack"],
                }
            ),
        )

    def test_attacked_client_map_colors_attackers_and_records_attack_types(self):
        selections = pd.DataFrame(
            {"round": [0, 1], "clients": [[0, 1], [1, 2]]}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "participation_histogram.png"
            with (
                patch("src.utils.logging_utils.plt.bar") as bar,
                patch("src.utils.logging_utils.plt.legend") as legend,
            ):
                summary = build_client_participation_histogram(
                    selections,
                    num_clients=3,
                    save_path=output_path,
                    client_attack_map={0: "no_attack", 1: "label_flip", 2: "ipm"},
                )

        self.assertEqual(
            bar.call_args.kwargs["color"],
            ["tab:blue", "tab:red", "tab:red"],
        )
        legend.assert_called_once()
        self.assertEqual(summary["rounds_selected"].tolist(), [1, 2, 1])
        self.assertEqual(summary["is_attacker"].tolist(), [False, True, True])
        self.assertEqual(
            summary["attack_type"].tolist(),
            ["no_attack", "label_flip", "ipm"],
        )


if __name__ == "__main__":
    unittest.main()
