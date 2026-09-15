import time

from hydra.utils import instantiate

from utils.data_utils import get_dataset_loader
from ..personalized_fixed.client import PersonalizedFixedClient


class FedRepFixedClient(PersonalizedFixedClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.head_local_epochs = client_args[2]
        self.representation_local_epochs = client_args[3]
        self.evaluation_only = False
        self.evaluation_result = None

    def create_pipe_commands(self):
        commands = super().create_pipe_commands()
        commands["client_state"] = self.load_client_state
        commands["evaluation_state"] = self.load_evaluation_state
        return commands

    def load_client_state(self, state_dict):
        self.model.load_state_dict(state_dict)

    def load_evaluation_state(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.evaluation_only = True

    def set_trainable_part(self, part):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith(f"{part}.")
        self._init_optimizer()

    def train(self):
        if self.evaluation_only:
            self.evaluate_aggregated_representation()
            return

        start = time.time()
        self.server_model_state = self.clone_model_state()

        self.local_epochs = self.head_local_epochs
        self.set_trainable_part("head")
        self.model_trainer.train_fn(self)
        local_head = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
            if key.startswith("head.")
        }
        loss, metrics = self.model_trainer.client_eval_fn(self)
        self.set_personalized_result(metrics, loss)

        self.local_epochs = self.representation_local_epochs
        self.set_trainable_part("representation")
        self.model_trainer.train_fn(self)
        self.get_grad()
        self.local_head = local_head
        self.result_time = time.time() - start

    def evaluate_aggregated_representation(self):
        start = time.time()
        self.local_epochs = self.head_local_epochs
        self.set_trainable_part("head")
        self.model_trainer.train_fn(self)
        validation_loss, validation_metrics = self.model_trainer.client_eval_fn(self)

        test_dataset = instantiate(
            self.cfg.test_dataset, cfg=self.cfg, mode="test", _recursive_=False
        )
        test_loader = get_dataset_loader(test_dataset, self.cfg, drop_last=False)
        validation_loader = self.valid_loader
        self.valid_loader = test_loader
        test_loss, test_metrics = self.model_trainer.client_eval_fn(self)
        self.valid_loader = validation_loader

        self.evaluation_result = {
            "rank": self.rank,
            "personalized_model": self.clone_model_state(),
            "validation_metrics": validation_metrics,
            "validation_loss": float(validation_loss),
            "test_metrics": test_metrics,
            "test_loss": float(test_loss),
        }
        self.result_time = time.time() - start

    def get_communication_content(self):
        if self.evaluation_result is not None:
            return self.evaluation_result
        result = super().get_communication_content()
        result["local_head"] = self.local_head
        return result
