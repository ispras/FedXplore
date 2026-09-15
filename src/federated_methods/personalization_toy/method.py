from types import SimpleNamespace

import numpy as np
import pandas as pd
from hydra.utils import instantiate

from ..fedavg.fedavg import FedAvg
from .client import PersonalizationToyClient
from .server import PersonalizationToyServer


class PersonalizationToy(FedAvg):
    def __init__(self, proximity):
        super().__init__()
        self.proximity = proximity

    def _init_server(self, cfg):
        self.server = PersonalizationToyServer(cfg)

    def _init_client_cls(self):
        assert (
            self.num_clients_subset == self.amount_of_clients
        ), "PersonalizationToy requires full client participation"
        super()._init_client_cls()
        self.client_cls = PersonalizationToyClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.append(self.proximity)

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        state = self.server.personalized_models[rank]
        content["personalized_model"] = state or {
            key: value.cpu()
            for key, value in self.server.global_model.state_dict().items()
        }
        return content

    def aggregate(self):
        aggregated_weights = super().aggregate()
        self._print_personalized_diagnostics()
        return aggregated_weights

    def _print_personalized_diagnostics(self):
        validation_metrics = (
            pd.concat(self.server.personalized_metrics).groupby(level=0).mean()
        )
        validation_loss = float(np.mean(self.server.personalized_losses))

        test_metrics = []
        test_losses = []
        for state in self.server.personalized_models:
            model = instantiate(
                self.cfg.model, num_classes=self.train_dataset.num_classes
            ).to(self.server.device)
            model.load_state_dict(state)
            context = SimpleNamespace(
                global_model=model,
                device=self.server.device,
                test_loader=self.server.test_loader,
                criterion=self.server.criterion,
            )
            targets, outputs, loss = self.server.model_trainer.server_eval_fn(context)
            test_metrics.append(
                self.server.model_trainer.calculate_metrics(targets, outputs)
            )
            test_losses.append(float(loss))

        mean_test_metrics = pd.concat(test_metrics).groupby(level=0).mean()
        mean_test_loss = float(np.mean(test_losses))
        validation_accuracy = float(validation_metrics.loc["Accuracy"].mean())
        test_accuracy = float(mean_test_metrics.loc["Accuracy"].mean())

        print("\nPersonalized Mean Validation Results:")
        print(validation_metrics)
        print(f"Personalized Mean Validation Loss: {validation_loss}")
        print("\nPersonalized Mean Unbiased Test Results:")
        print(mean_test_metrics)
        print(f"Personalized Mean Unbiased Test Loss: {mean_test_loss}")
        print(
            "PERSONALIZATION_DIAGNOSTIC "
            f"round={self.cur_round} "
            f"validation_accuracy={validation_accuracy:.6f} "
            f"test_accuracy={test_accuracy:.6f} "
            f"gap={validation_accuracy - test_accuracy:.6f}"
        )
