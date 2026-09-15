import copy
import time
from collections import OrderedDict

from hydra.utils import instantiate

from ..personalized.client import PersonalizedClient


class pFedMeClient(PersonalizedClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.proximity = client_args[2]
        self.k_steps = client_args[3]
        self.personal_learning_rate = client_args[4]
        self.outer_learning_rate = self.cfg.optimizer.lr
        self.outer_reference = copy.deepcopy(self.model)

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        loss += (
            0.5
            * self.proximity
            * sum(
                (parameter - reference.detach()).norm() ** 2
                for parameter, reference in zip(
                    self.model.parameters(),
                    self.outer_reference.parameters(),
                )
            )
        )
        return loss

    def inner_solve(self, inputs, targets):
        self.optimizer = instantiate(
            self.cfg.optimizer,
            params=self.model.parameters(),
            lr=self.personal_learning_rate,
        )
        for _ in range(self.k_steps):
            self.optimizer.zero_grad()
            loss = self.get_loss_value(self.model(inputs), targets)
            loss.backward()
            self.optimizer.step()

    def update_outer_state(self, outer_state):
        theta_state = self.clone_model_state()
        factor = self.outer_learning_rate * self.proximity
        return {
            key: (
                value - factor * (value - theta_state[key])
                if value.is_floating_point()
                else theta_state[key]
            )
            for key, value in outer_state.items()
        }

    def train(self):
        start = time.time()
        global_state = self.clone_model_state()
        self.server_model_state = global_state
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )
        outer_state = copy.deepcopy(global_state)

        for _ in range(self.local_epochs):
            for _, (inputs, targets) in self.train_loader:
                inputs = inputs[0].to(self.device)
                targets = targets.to(self.device)
                self.model.load_state_dict(outer_state)
                self.outer_reference.load_state_dict(outer_state)
                self.inner_solve(inputs, targets)
                outer_state = self.update_outer_state(outer_state)

        theta_state = self.clone_model_state()
        loss, metrics = self.model_trainer.client_eval_fn(self)
        self.set_personalized_result(metrics, loss)
        self.personalized_model_state = theta_state

        self.model.load_state_dict(outer_state)
        self.grad = OrderedDict(
            (key, value.detach().cpu() - global_state[key])
            for key, value in self.model.state_dict().items()
        )
        self.result_time = time.time() - start
