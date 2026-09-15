from types import SimpleNamespace

import numpy as np
import pandas as pd
from hydra.utils import instantiate

from ..fedavg.fedavg import FedAvg
from .server import PersonalizedFixedServer


class PersonalizedFixedMethod(FedAvg):
    def _init_server(self, cfg):
        self.server = PersonalizedFixedServer(cfg)

    def _init_client_cls(self):
        assert (
            self.num_clients_subset == self.amount_of_clients
        ), f"{type(self).__name__} requires full client participation"
        super()._init_client_cls()

    def get_personalized_state(self, rank):
        state = self.server.personalized_models[rank]
        return state or {
            key: value.detach().cpu().clone()
            for key, value in self.server.global_model.state_dict().items()
        }

    def print_personalized_diagnostics(self):
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

        self.print_diagnostic_metrics(
            self.server.personalized_metrics,
            self.server.personalized_losses,
            test_metrics,
            test_losses,
        )

    def print_diagnostic_metrics(
        self, validation_metrics, validation_losses, test_metrics, test_losses
    ):
        mean_validation = pd.concat(validation_metrics).groupby(level=0).mean()
        mean_test = pd.concat(test_metrics).groupby(level=0).mean()
        validation_loss = float(np.mean(validation_losses))
        test_loss = float(np.mean(test_losses))
        validation_accuracy = float(mean_validation.loc["Accuracy"].mean())
        test_accuracy = float(mean_test.loc["Accuracy"].mean())

        print("\nPersonalized Mean Validation Results:")
        print(mean_validation)
        print(f"Personalized Mean Validation Loss: {validation_loss}")
        print("\nPersonalized Mean Unbiased Test Results:")
        print(mean_test)
        print(f"Personalized Mean Unbiased Test Loss: {test_loss}")
        print(
            "PERSONALIZATION_DIAGNOSTIC "
            f"method={type(self).__name__} "
            f"round={self.cur_round} "
            f"validation_accuracy={validation_accuracy:.6f} "
            f"test_accuracy={test_accuracy:.6f} "
            f"gap={validation_accuracy - test_accuracy:.6f}"
        )
