from ..personalized.client import PerClient
import time
import copy
from hydra.utils import instantiate


class DittoClient(PerClient):
    def __init__(
        self,
        *client_args,
        **client_kwargs,
    ):
        base_client_args = client_args[:2]
        super().__init__(*base_client_args, **client_kwargs)
        self.proximity = client_args[2]
        self.client_args = client_args

        self.do_proximity = True
        self.server_model = copy.deepcopy(self.model)
        self.local_model_state = None

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["update_model"] = (
            lambda state_dict: self.model.load_state_dict(
                {k: v.to(self.device) for k, v in state_dict.items()}
            )
        )
        pipe_commands_map["local_model"] = self.set_local_state

        return pipe_commands_map

    def set_local_state(self, state_dict):
        self.local_model_state = state_dict

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        if self.do_proximity:
            proximity = (
                0.5
                * self.proximity
                * sum(
                    [
                        (p.float() - q.float().detach()).norm() ** 2
                        for (_, p), (_, q) in zip(
                            self.model.named_parameters(),
                            self.server_model.named_parameters(),
                        )
                    ]
                )
            )
            loss += proximity
        return loss

    def train(self):
        start = time.time()
        self.server_model_state = copy.deepcopy(self.model).state_dict()

        # Train server model
        self.do_proximity = False
        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )
        self._init_optimizer()
        self.model_trainer.train_fn(self)

        # Upload trained weights to server model for regularization
        self.server_model.load_state_dict(self.model.state_dict())

        # Evaluate local model before training
        self.model.load_state_dict(self.local_model_state)
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )

        # Train local model
        self.do_proximity = True
        self._init_optimizer()
        self.model_trainer.train_fn(self)
        self.local_model_state = self.model.state_dict()

        # Load server weight to model for get_grad
        self.model.load_state_dict(self.server_model.state_dict())

        self.get_grad()
        self.result_time = time.time() - start

    def get_communication_content(self):
        result_dict = super().get_communication_content()
        result_dict["local_model"] = {
            k: v.cpu() for k, v in self.local_model_state.items()
        }
        return result_dict
