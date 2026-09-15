import copy
import time

from ..personalized_fixed.client import PersonalizedFixedClient


class FedAMPFixedClient(PersonalizedFixedClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.proximity = client_args[2]
        self.relative_model = copy.deepcopy(self.model)

    def create_pipe_commands(self):
        commands = super().create_pipe_commands()
        commands["client_model"] = self.set_client_model
        commands["relative_model"] = self.set_relative_model
        return commands

    def set_client_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def set_relative_model(self, state_dict):
        self.relative_model.load_state_dict(state_dict)

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        loss += (
            0.5
            * self.proximity
            * sum(
                (parameter - reference.detach()).norm() ** 2
                for parameter, reference in zip(
                    self.model.parameters(), self.relative_model.parameters()
                )
            )
        )
        return loss

    def train(self):
        start = time.time()
        self.server_model_state = self.clone_model_state()
        self._init_optimizer()
        self.model_trainer.train_fn(self)
        loss, metrics = self.model_trainer.client_eval_fn(self)
        self.set_personalized_result(metrics, loss)
        self.grad = self.clone_model_state()
        self.result_time = time.time() - start
