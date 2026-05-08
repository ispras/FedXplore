import copy
import random
import torch
import pandas as pd

from ..byzantine_base.byzantine import ByzantineBase
from .byz_vr_marina_pp_client import ByzVRMarinaPPClient


class ByzVRMarinaPP(ByzantineBase):
    def __init__(self, p, global_lr, lambda_, aggregation_type, bucket_size):
        super().__init__()
        self.p = p
        self.global_lr = global_lr
        self.lambda_ = lambda_
        self.aggregation_type = aggregation_type
        self.bucket_size = bucket_size

    def _init_federated(self, cfg):
        super()._init_federated(cfg)
        self.batch_idxs = [0] * self.amount_of_clients
        self.save_prev_global_model_state = None

    def _init_client_cls(self):
        assert "SGD" in str(
            self.cfg.optimizer._target_
        ), f"ByzVRMarinaPP works only with SGD client optimizer. You provide: {self.cfg.optimizer._target_}"

        super()._init_client_cls()
        self.client_cls = ByzVRMarinaPPClient
        self.client_kwargs["client_cls"] = self.client_cls

    def train_round(self):
        self.flag_full_grad = bool(random.random() < self.p)
        if self.cur_round == 0:
            self.flag_full_grad = True
        print(f"[Method]: cur c_k = {int(self.flag_full_grad)}")

        return super().train_round()

    def aggregate(self):
        print(f"[Method]: batch idxs\n{self.batch_idxs}")
        grads = [self.server.client_gradients[r] for r in self.list_clients]

        # Clipping gradients
        clipped_grads = []
        for g in grads:
            clipped_grads.append(self.clip_grad_dict(g))

        # ---------- Robust Aggregation ----------
        if self.aggregation_type == "CM":
            agg_grad = coordinate_median(clipped_grads)
        elif self.aggregation_type == "RFA":
            buckets = bucketize(clipped_grads, self.bucket_size)
            bucket_aggs = [rfa_aggregate(b) for b in buckets]
            agg_grad = rfa_aggregate(bucket_aggs)
        else:
            raise ValueError("Unknown aggregation")

        # ---------- Model update ----------
        aggregated_weights = copy.deepcopy(self.server.global_model.state_dict())
        self.prev_global_model_state = copy.deepcopy(aggregated_weights)
        self.prev_agg_grads = copy.deepcopy(agg_grad)

        for k in aggregated_weights:
            aggregated_weights[k] -= self.global_lr * agg_grad[k].to(self.server.device)

        return aggregated_weights

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

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        if self.cur_round == 0:
            content["prev_global_model_state"] = copy.deepcopy(
                self.server.global_model.state_dict()
            )
            content["prev_agg_grad"] = copy.deepcopy(
                self.server.global_model.state_dict()
            )
        else:
            content["prev_global_model_state"] = self.prev_global_model_state
            content["prev_agg_grad"] = self.prev_agg_grads

        content["cur_batch_idx"] = self.batch_idxs[rank]
        content["flag_full_grad"] = self.flag_full_grad

        self.batch_idxs[rank] += 1
        return content


def coordinate_median(grad_list):
    """
    grad_list: List[Dict[param_name -> tensor]]
    """
    aggregated = {}
    for key in grad_list[0].keys():
        stacked = torch.stack([g[key] for g in grad_list], dim=0)
        aggregated[key] = torch.median(stacked, dim=0).values
    return aggregated


def flatten_grad(grad):
    return torch.cat([v.view(-1) for v in grad.values()])


def unflatten_grad(flat, reference):
    out = {}
    idx = 0
    for k, v in reference.items():
        numel = v.numel()
        out[k] = flat[idx : idx + numel].view_as(v)
        idx += numel
    return out


def rfa_aggregate(grad_list, iters=10, eps=1e-6):
    flats = [flatten_grad(g) for g in grad_list]
    flats = torch.stack(flats)

    # init = mean
    z = flats.mean(dim=0)

    for _ in range(iters):
        weights = torch.norm(flats - z, dim=1)
        weights = torch.clamp(weights, min=eps)
        z = (flats / weights.unsqueeze(1)).sum(dim=0) / (1.0 / weights).sum()

    return unflatten_grad(z, grad_list[0])


def bucketize(grads, bucket_size):
    random.shuffle(grads)
    return [grads[i : i + bucket_size] for i in range(0, len(grads), bucket_size)]
