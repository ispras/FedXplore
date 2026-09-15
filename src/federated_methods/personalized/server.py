import os
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate

from utils.metrics_utils import stopping_criterion
from utils.utils import create_model_info
from ..fedavg.fedavg_server import FedAvgServer


class PersonalizedServer(FedAvgServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        count = cfg.federated_params.amount_of_clients
        self.personalized_models = [OrderedDict() for _ in range(count)]
        self.personalized_metrics = [None for _ in range(count)]
        self.personalized_losses = [0.0 for _ in range(count)]
        self.personalized_validation_metrics = None
        self.personalized_validation_loss = None
        self.personalized_test_metrics = None
        self.personalized_test_loss = None

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        rank = client_result["rank"]
        metrics, loss, _ = client_result["personalized_metrics"]
        self.personalized_models[rank] = client_result["personalized_model"]
        self.personalized_metrics[rank] = metrics
        self.personalized_losses[rank] = loss

    def save_best_model(self, round):
        # Personalized checkpoints are selected after round aggregation.
        return

    def evaluate_personalized_models(self):
        test_metrics = []
        test_losses = []
        for state in self.personalized_models:
            model = instantiate(
                self.cfg.model,
                num_classes=self.test_df.num_classes,
            ).to(self.device)
            model.load_state_dict(state)
            context = SimpleNamespace(
                global_model=model,
                device=self.device,
                test_loader=self.test_loader,
                criterion=self.criterion,
            )
            targets, outputs, loss = self.model_trainer.server_eval_fn(context)
            test_metrics.append(
                self.model_trainer.calculate_metrics(targets, outputs)
            )
            test_losses.append(float(loss))

        self.set_personalized_evaluation(
            self.personalized_metrics,
            self.personalized_losses,
            test_metrics,
            test_losses,
        )

    def set_personalized_evaluation(
        self,
        validation_metrics,
        validation_losses,
        test_metrics,
        test_losses,
    ):
        self.personalized_validation_metrics = (
            pd.concat(validation_metrics).groupby(level=0).mean()
        )
        self.personalized_validation_loss = float(np.mean(validation_losses))
        self.personalized_test_metrics = (
            pd.concat(test_metrics).groupby(level=0).mean()
        )
        self.personalized_test_loss = float(np.mean(test_losses))

        print("\nPersonalized Mean Validation Results:")
        print(self.personalized_validation_metrics)
        print(
            "Personalized Mean Validation Loss: "
            f"{self.personalized_validation_loss}"
        )
        print("\nPersonalized Mean Unbiased Test Results:")
        print(self.personalized_test_metrics)
        print(f"Personalized Mean Unbiased Test Loss: {self.personalized_test_loss}")

    def save_best_personalized_model(self, round):
        if not self.best_metrics:
            return

        states = {
            rank: {
                key: value.detach().cpu().clone()
                for key, value in state.items()
            }
            for rank, state in enumerate(self.personalized_models)
        }
        rounds_no_improve, best_metrics = stopping_criterion(
            self.personalized_validation_loss,
            self.personalized_validation_metrics,
            self.best_metrics,
            rounds_no_improve=self.rounds_no_improve,
        )
        self.rounds_no_improve = rounds_no_improve
        if rounds_no_improve != 0 or np.isnan(self.personalized_validation_loss):
            return

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        self.best_metrics = best_metrics
        self.best_round = round
        self.checkpoint_path = f"{self.model_path}_round_{round}.pt"
        model_info = create_model_info(
            model_state=states,
            valid_metrics=self.personalized_validation_metrics,
            valid_loss=self.personalized_validation_loss,
            test_metrics=self.personalized_test_metrics,
            test_loss=self.personalized_test_loss,
            cfg=self.cfg,
        )
        torch.save(model_info, self.checkpoint_path)
