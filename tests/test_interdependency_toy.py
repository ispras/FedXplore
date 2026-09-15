from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import torch

from src.attack_clients import BinaryLabelFlipClient
from src.federated_datasets.synthetic_2d_dataset import make_gaussian_2d_dataframe
from src.model_trainers.image_trainer import ImageTrainer
from src.utils.model_utils import logistic_regression
from ui.launcher import load_templates


class InterdependencyToyTests(unittest.TestCase):
    def test_synthetic_data_is_balanced_and_reproducible(self) -> None:
        first = make_gaussian_2d_dataframe(200, 2.0, 1.0, 42)
        second = make_gaussian_2d_dataframe(200, 2.0, 1.0, 42)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["target"].value_counts().to_dict(), {0: 100, 1: 100})
        self.assertGreater(
            first.loc[first.target == 1, ["x1", "x2"]].mean().min(),
            first.loc[first.target == 0, ["x1", "x2"]].mean().max(),
        )

    def test_logistic_regression_accepts_2d_features(self) -> None:
        model = logistic_regression(input_size=2, num_classes=2)

        self.assertEqual(model(torch.zeros(4, 2)).shape, (4, 2))

    def test_image_metrics_use_dataset_agnostic_column(self) -> None:
        metrics = ImageTrainer(None).calculate_metrics(
            fin_targets=[0, 1],
            fin_outputs=[[1.0, 0.0], [0.0, 1.0]],
        )

        self.assertEqual(metrics.columns.tolist(), ["overall"])

    def test_binary_label_flip_swaps_both_classes(self) -> None:
        data = pd.DataFrame({"target": [0, 1, 1, 0]})

        flipped = BinaryLabelFlipClient()._flip_labels(data)

        self.assertEqual(flipped.target.tolist(), [1, 0, 0, 1])

    def test_ui_has_a_template_for_each_selector(self) -> None:
        templates = load_templates(Path("ui/templates"))

        selectors = {
            templates[key].form["client_selector"]
            for key in (
                "interdependency_toy",
                "interdependency_toy_uniform",
                "interdependency_toy_fedcbs",
            )
        }
        self.assertEqual(selectors, {"pow", "uniform", "fedcbs"})


if __name__ == "__main__":
    unittest.main()
