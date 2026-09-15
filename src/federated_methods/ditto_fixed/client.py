import copy
import time

from ..personalized_fixed.client import PersonalizedFixedClient


class DittoFixedClient(PersonalizedFixedClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.proximity = client_args[2]
        self.global_reference = copy.deepcopy(self.model)
        self.personalizing = False

    def create_pipe_commands(self):
        commands = super().create_pipe_commands()
        commands["personalized_model"] = self.set_personalized_model
        return commands

    def set_personalized_model(self, state_dict):
        self.personalized_model_state = state_dict

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        if self.personalizing:
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
        global_state = self.clone_model_state()
        self.server_model_state = global_state

        self.personalizing = False
        self._init_optimizer()
        self.model_trainer.train_fn(self)
        trained_global_state = self.clone_model_state()

        self.model.load_state_dict(self.personalized_model_state)
        self.global_reference.load_state_dict(global_state)
        self.personalizing = True
        self._init_optimizer()
        self.model_trainer.train_fn(self)
        loss, metrics = self.model_trainer.client_eval_fn(self)
        self.set_personalized_result(metrics, loss)

        self.model.load_state_dict(trained_global_state)
        self.get_grad()
        self.result_time = time.time() - start
