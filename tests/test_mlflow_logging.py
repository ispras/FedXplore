from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.utils.logging_utils import BaseLogger, MLFlowLogger


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


if __name__ == "__main__":
    unittest.main()
