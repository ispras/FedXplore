from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from federated_methods.ditto.ditto import Ditto
from federated_methods.fedamp.fedamp import FedAMP
from federated_methods.fedavg.fedavg_server import FedAvgServer
from federated_methods.fedrep.fedrep import FedRep
from federated_methods.personalized.server import PersonalizedServer
from federated_methods.pfedme.pfedme import pFedMe


def metric_frame(value):
    return pd.DataFrame({"overall": [value]}, index=["Accuracy"])


class PersonalizedMethodTests(unittest.TestCase):
    def test_empty_saving_metrics_keeps_validation_without_checkpoint(self):
        server = FedAvgServer.__new__(FedAvgServer)
        server.server_metrics = [metric_frame(0.5), metric_frame(0.75)]
        server.server_losses = [0.2, 0.4]
        server.server_val_df_len = [2, 2]
        server.metric_aggregation = "uniform"
        server.best_metrics = {}
        server.global_model = SimpleNamespace(
            state_dict=lambda: {"weight": torch.ones(1)}
        )
        server.last_metrics = metric_frame(0.6)
        server.test_loss = 0.3
        server.checkpoint_path = None

        with patch("federated_methods.fedavg.fedavg_server.torch.save") as save:
            server.save_best_model(0)

        self.assertAlmostEqual(server.latest_validation_loss, 0.3)
        self.assertAlmostEqual(
            float(server.latest_validation_metrics.loc["Accuracy", "overall"]),
            0.625,
        )
        self.assertIsNone(server.checkpoint_path)
        save.assert_not_called()

    def test_personalized_checkpoint_uses_its_own_states_and_metrics(self):
        server = PersonalizedServer.__new__(PersonalizedServer)
        first = {"weight": torch.tensor([1.0])}
        second = {"weight": torch.tensor([2.0])}
        server.personalized_models = [first, second]
        server.personalized_validation_metrics = metric_frame(0.8)
        server.personalized_validation_loss = 0.25
        server.personalized_test_metrics = metric_frame(0.7)
        server.personalized_test_loss = 0.35
        server.best_metrics = {"loss": 1000}
        server.rounds_no_improve = 0
        server.best_round = 0
        server.checkpoint_path = None
        server.model_path = "/tmp/personalized-model"
        server.cfg = SimpleNamespace()

        model_info = {}

        def make_model_info(**kwargs):
            model_info.update(kwargs)
            return kwargs

        with (
            patch(
                "federated_methods.personalized.server.create_model_info",
                side_effect=make_model_info,
            ),
            patch("federated_methods.personalized.server.torch.save") as save,
        ):
            server.save_best_personalized_model(4)

        self.assertEqual(set(model_info["model_state"]), {0, 1})
        self.assertEqual(model_info["valid_loss"], 0.25)
        self.assertEqual(model_info["test_loss"], 0.35)
        self.assertIs(
            model_info["valid_metrics"],
            server.personalized_validation_metrics,
        )
        self.assertIs(model_info["test_metrics"], server.personalized_test_metrics)
        first["weight"].fill_(9.0)
        self.assertEqual(model_info["model_state"][0]["weight"].item(), 1.0)
        save.assert_called_once()

    def test_canonical_hydra_names_create_migrated_implementations(self):
        configs = Path("src/configs/federated_method")
        expected = {
            "ditto": Ditto,
            "pfedme": pFedMe,
            "fedrep": FedRep,
            "fedamp": FedAMP,
        }
        for name, method_class in expected.items():
            method = instantiate(OmegaConf.load(configs / f"{name}.yaml"))
            self.assertIs(type(method), method_class)

        for name in expected:
            self.assertFalse((configs / f"{name}_fixed.yaml").exists())

    def test_fedavg_and_personalized_expose_the_same_metric_splits(self):
        global_server = SimpleNamespace(
            test_loss=0.3,
            last_metrics=metric_frame(0.7),
            latest_validation_loss=0.2,
            latest_validation_metrics=metric_frame(0.8),
        )
        personalized_server = SimpleNamespace(
            personalized_test_loss=0.35,
            personalized_test_metrics=metric_frame(0.65),
            personalized_validation_loss=0.25,
            personalized_validation_metrics=metric_frame(0.75),
        )

        from federated_methods.fedavg.fedavg import FedAvg
        from federated_methods.personalized.method import PersonalizedMethod

        fedavg = FedAvg()
        fedavg.server = global_server
        personalized = PersonalizedMethod()
        personalized.server = personalized_server

        class RecordingLogger:
            def __init__(self):
                self.names = set()

            def log_scalar(self, value, name, cur_round):
                self.names.add(name)

            def log_pandas(self, frame, group_name, cur_round):
                self.names.update(
                    f"{group_name}{row}_{column}"
                    for row in frame.index
                    for column in frame.columns
                )

        expected_names = {
            "val/loss",
            "val/Accuracy_overall",
            "test/loss",
            "test/Accuracy_overall",
        }
        for method in (fedavg, personalized):
            method.cur_round = 2
            method.logger = RecordingLogger()
            method.log_evaluation_metrics()
            self.assertEqual(method.logger.names, expected_names)


if __name__ == "__main__":
    unittest.main()
