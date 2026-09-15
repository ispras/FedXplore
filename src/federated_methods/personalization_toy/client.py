import copy
import time

from ..fedavg.fedavg_client import FedAvgClient


class PersonalizationToyClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args[:2], **client_kwargs)
        self.client_args = client_args
        self.proximity = client_args[2]
        self.global_reference = copy.deepcopy(self.model)
        self.personalized_model_state = None
        self.personalizing = False

    def create_pipe_commands(self):
        commands = super().create_pipe_commands()
        commands["personalized_model"] = self.set_personalized_model
        return commands

    def set_personalized_model(self, state_dict):
        self.personalized_model_state = state_dict

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        if self.personalizing and self.proximity:
            loss += (
                0.5
                * self.proximity
                * sum(
                    (parameter - reference.detach()).norm() ** 2
                    for parameter, reference in zip(
                        self.model.parameters(), self.global_reference.parameters()
                    )
                )
            )
        return loss

    def train(self):
        start = time.time()
        global_state = copy.deepcopy(self.model.state_dict())
        self.server_model_state = global_state

        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )
        self.personalizing = False
        self._init_optimizer()
        self.model_trainer.train_fn(self)
        trained_global_state = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.personalized_model_state)
        self.global_reference.load_state_dict(global_state)
        self.personalizing = True
        self._init_optimizer()
        self.model_trainer.train_fn(self)
        personalized_loss, personalized_metrics = self.model_trainer.client_eval_fn(
            self
        )
        self.personalized_model_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

        self.model.load_state_dict(trained_global_state)
        self.get_grad()
        self.personalized_result = (
            personalized_metrics,
            personalized_loss,
            len(self.valid_dataset),
        )
        self.result_time = time.time() - start

    def get_communication_content(self):
        result = super().get_communication_content()
        result["personalized_model"] = self.personalized_model_state
        result["personalized_metrics"] = self.personalized_result
        return result
