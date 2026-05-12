import time
import copy
import torch
import random
from itertools import islice

from ..fedavg.fedavg_client import FedAvgClient


class ByzVRMarinaPPClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]  # `cfg` and `df`
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.batch_idx = None
        self.flag_full_grad = None
        self.prev_agg_grad = None
        self.prev_global_model_state = None
        self.lambda_ = self.cfg.federated_method.lambda_

    def get_full_gradient(self):
        batch = next(
            islice(
                self.train_loader,
                self.batch_idx % len(self.train_loader),
                (self.batch_idx % len(self.train_loader)) + 1,
            )
        )

        self.model.train()
        _, (input, targets) = batch
        inp = input[0].to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inp)

        loss = self.get_loss_value(outputs, targets)
        loss.backward()

        self.optimizer.step()

        inp = input[0].to("cpu")
        targets = targets.to("cpu")

    def compute_gradient_single_sample(self, model_state, x_j, y_j):
        self.model.load_state_dict(model_state)
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(x_j)
        loss = self.get_loss_value(outputs, y_j)
        loss.backward()

        grad = copy.deepcopy(self.server_model_state)
        for name, tensor in grad.items():
            param = dict(self.model.named_parameters()).get(name, None)
            if param is not None and param.grad is not None:
                grad[name] = param.grad.detach().clone().to("cpu")
            else:
                grad[name] = torch.zeros_like(tensor, device="cpu")

        return grad

    def get_approx_gradient(self):
        # get batch
        cur_batch_idx = random.randrange(1, len(self.train_loader) - 1)
        batch = next(
            islice(
                self.train_loader,
                cur_batch_idx,
                cur_batch_idx + 1,
            )
        )
        _, (inputs, targets) = batch

        # inputs: (B, C, H, W) or (B, D)
        inputs = inputs[0].to(self.device)
        targets = targets.to(self.device)

        # get random example
        j = random.randrange(inputs.shape[0])
        x_j = inputs[j].unsqueeze(0)  # shape (1, ...)
        y_j = targets[j].unsqueeze(0)

        # ∇f_j(x^k)
        grad_cur = self.compute_gradient_single_sample(
            self.server_model_state,
            x_j,
            y_j,
        )

        # ∇f_j(x^{k-1})
        grad_prev = self.compute_gradient_single_sample(
            self.prev_global_model_state,
            x_j,
            y_j,
        )

        # clip (∇f_j(x^k) - ∇f_j(x^{k-1}))
        diff = {}
        for k in grad_cur.keys():
            diff[k] = grad_cur[k] - grad_prev[k]

        diff = self.clip_grad_dict(diff)

        # g_i^k = g_i^{k-1} + clip(grad_cur - grad_prev)
        final_grad = copy.deepcopy(self.server_model_state)
        for k in final_grad.keys():
            final_grad[k] = self.prev_agg_grad[k] + diff[k]

        self.grad = final_grad

    def clip_grad_dict(self, grad_dict):
        total_norm_sq = 0.0

        for v in grad_dict.values():
            if torch.is_floating_point(v):
                total_norm_sq += v.norm() ** 2

        total_norm = torch.sqrt(total_norm_sq)

        if total_norm == 0 or total_norm <= self.lambda_:
            return grad_dict

        scale = self.lambda_ / total_norm

        clipped = {}
        for k, v in grad_dict.items():
            if torch.is_floating_point(v):
                clipped[k] = v * scale
            else:
                clipped[k] = v

        return clipped

    def train(self):
        start = time.time()
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
            self
        )

        # ---------- BYZ-VR-MARINA-PP ---------- #
        if self.flag_full_grad:
            self.get_full_gradient()
            self.get_grad()
        else:
            assert self.prev_global_model_state.keys() == self.server_model_state.keys()
            self.get_approx_gradient()
        # ---------- BYZ-VR-MARINA-PP ---------- #
        assert self.grad.keys() == self.server_model_state.keys()

        if self.print_metrics:
            self.client_val_loss, self.client_metrics = (
                self.model_trainer.client_eval_fn(self)
            )
        self.result_time = time.time() - start

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["cur_batch_idx"] = self.set_cur_batch_idx
        pipe_commands_map["flag_full_grad"] = self.set_flag_full_grad
        pipe_commands_map["prev_global_model_state"] = self.set_prev_global_model_state
        pipe_commands_map["prev_agg_grad"] = self.set_prev_agg_grad
        return pipe_commands_map

    def set_cur_batch_idx(self, new_batch_idx):
        self.batch_idx = new_batch_idx

    def set_flag_full_grad(self, flag_full_grad):
        self.flag_full_grad = flag_full_grad

    def set_prev_global_model_state(self, prev_global_model_state):
        self.prev_global_model_state = prev_global_model_state

    def set_prev_agg_grad(self, prev_agg_grad):
        self.prev_agg_grad = prev_agg_grad
