import copy
import time
import torch
import torch.nn as nn
from hydra.utils import instantiate
from ..personalized.client import PerClient


class FedRepClient(PerClient):
    def __init__(
        self,
        *client_args,
        **client_kwargs,
    ):
        super().__init__(*client_args, **client_kwargs)
        self.warmup = True

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["update_model"] = self.load_body_new_head
        pipe_commands_map["warmup"] = self.set_wp
        return pipe_commands_map

    def set_wp(self, warmup_flag):
        self.warmup = warmup_flag

    def load_body_new_head(self, server_state_dict):
        new_state_dict = copy.deepcopy(server_state_dict)

        # Hardcode for our Resnet
        for k, v in new_state_dict.items():
            # Reinit head
            if ("linear" in k) and (not self.warmup):
                linear_layer = nn.Linear(
                    512 * self.model.block.expansion, self.df.num_classes
                )

                with torch.no_grad():
                    if "weight" in k:
                        new_state_dict[k] = linear_layer.weight
                    elif "bias" in k:
                        new_state_dict[k] = linear_layer.bias

        self.model.load_state_dict(new_state_dict)

    def freeze_model(self, freeze_mode="unfreeze"):
        head_require_grad = freeze_mode != "head"
        body_require_grad = freeze_mode != "body"

        for name, param in self.model.named_parameters():
            if "linear" in name:
                param.requires_grad = head_require_grad
            else:
                param.requires_grad = body_require_grad

        self._init_optimizer()

    def train(self):
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        start = time.time()

        # ---------- FedREP ---------- #
        if self.warmup:
            # Evaluate server model
            self.server_val_loss, self.server_metrics = (
                self.model_trainer.client_eval_fn(self)
            )
            # Just training server model
            self.model_trainer.train_fn(self)
        else:
            # Training head
            self.freeze_model(freeze_mode="body")  # freeze body, unfreeze head
            self.model_trainer.train_fn(self)

            # Evaluate personalized model
            self.server_val_loss, self.server_metrics = (
                self.model_trainer.client_eval_fn(self)
            )

            # Training feature extractor
            self.local_epochs = 1
            self.freeze_model(freeze_mode="head")  # freeze head, unfreeze body
            self.model_trainer.train_fn(self)

        # ---------- FedREP ---------- #

        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )

        self.get_grad()
        # Save training time
        self.result_time = time.time() - start
