import math
import torch
from collections import OrderedDict

from ..byzantine_base.byzantine import ByzantineBase
from .safeguard_server import Safeguard_server


class Safeguard(ByzantineBase):
    """
    Byzantine-Resilient Non-Convex Stochastic Gradient Descent:
    https://arxiv.org/pdf/2012.14368
    """

    def __init__(
        self, T_0, T_1, multiplier, min_thresh_A, min_thresh_B, noise_std_coef, lr
    ):
        super().__init__()
        self.thresholds = (T_0, T_1)
        self.multiplier = multiplier
        self.min_thresh = (min_thresh_A, min_thresh_B)
        self.noise_std_coef = noise_std_coef
        self.lr = lr

    def _init_server(self, cfg):
        super()._init_server(cfg)
        self.server = Safeguard_server(cfg)

    def reset_accumulation(self, index):
        if self.cur_round % self.thresholds[index] == 0:
            for i in range(self.cfg.federated_params.amount_of_clients):
                for key in self._param_names:
                    self.server.client_safeguards[index][i][key] = 0

    def accumulate_gradients(self, index):
        for rank in self.list_clients:
            for key in self._param_names:
                self.server.client_safeguards[index][rank][
                    key
                ] += self.server.client_gradients[rank][key] / len(
                    self.server.good_clients
                )

    def finding_median_grads(self, index):
        client_scores = []
        for rank in self.list_clients:
            scores = []
            for other_rank in self.list_clients:
                if rank == other_rank:
                    continue
                scores.append(
                    (
                        other_rank,
                        self.diff_norm(
                            self.server.client_safeguards[index][rank],
                            self.server.client_safeguards[index][other_rank],
                        ),
                    )
                )
            scores.sort(key=lambda dist: dist[1])
            # THIS IS INCORRECT, MINUS SHOULD BE PLUS
            median_index = math.ceil(len(scores) / 2 - 1)
            client_scores.append((rank, scores[median_index]))

        client_scores.sort(key=lambda dist: dist[1])
        self.server.min_score[index] = client_scores[0][1]
        min_rank = client_scores[0][0]
        self.server.grad_med_acum[index] = self.server.client_safeguards[index][
            min_rank
        ]

    def diff_norm(self, grad1, grad2):
        tmp_grad = OrderedDict()
        for key, _ in self.server.global_model.named_parameters():
            tmp_grad[key] = grad1[key] - grad2[key]
        tmp_grad_flat = torch.cat([x.flatten() for x in list(tmp_grad.values())])
        return torch.norm(tmp_grad_flat)

    def filter_workers(self, index):
        threshold = self.multiplier * min(
            self.min_thresh[index], self.server.min_score[index][1]
        )
        good_clients = []
        for rank in self.list_clients:
            dist = self.diff_norm(
                self.server.client_safeguards[index][rank],
                self.server.grad_med_acum[index],
            )
            if dist < 2 * threshold:
                good_clients.append(rank)
        return good_clients

    def add_noise(self, aggregated_weights):
        for key, weights in aggregated_weights.items():
            gaussian_noise = torch.normal(
                0,
                self.noise_std_coef,
                size=weights.shape,
            ).to(self.server.device)
            aggregated_weights[key] = weights - self.lr * gaussian_noise
        return aggregated_weights

    def aggregate(self):
        self._param_names = [
            name for name, _ in self.server.global_model.named_parameters()
        ]
        self._total_dim = sum(
            p.numel() for _, p in self.server.global_model.named_parameters()
        )
        self.server._param_names = self._param_names
        self.server._total_dim = self._total_dim
        self.make_pre_aggregation()

        if self.cur_round == 0:
            for i in self.server.good_clients:
                for key in self._param_names:
                    self.server.client_safeguards[0][i][key] = 0
                    self.server.client_safeguards[1][i][key] = 0
        # 2) Two‐phase safeguard
        all_medians = []  # to record per-phase medians
        for phase in (0, 1):
            # reset & accumulate in-place (unchanged)
            self.reset_accumulation(phase)
            self.accumulate_gradients(phase)

            # vectorize median search
            mat = self._stack_client_safeguards(phase)  # C×D
            median_vals, median_idx = self._find_medians(mat)  # both length C

            # store min_score and grad_med_acum
            # pick the client with the smallest of those median_vals
            min_row = torch.argmin(median_vals).item()
            self.server.min_score[phase] = (
                self.list_clients[min_row],
                median_vals[min_row].item(),
            )
            # the accumulated gradient of that “median” client
            chosen_rank = self.list_clients[median_idx[min_row].item()]
            self.server.grad_med_acum[phase] = self.server.client_safeguards[phase][
                chosen_rank
            ]

        # 3) vectorized filtering
        # build phase0 and phase1 mats
        mat0 = self._stack_client_safeguards(0)
        med0 = torch.cat(
            [self.server.grad_med_acum[0][k].view(-1) for k in self.server._param_names]
        ).to(self.server.device)
        d0 = torch.norm(mat0 - med0.unsqueeze(0), dim=1)  # length C

        mat1 = self._stack_client_safeguards(1)
        med1 = torch.cat(
            [self.server.grad_med_acum[1][k].view(-1) for k in self.server._param_names]
        ).to(self.server.device)
        d1 = torch.norm(mat1 - med1.unsqueeze(0), dim=1)

        # thresholds
        th0 = self.multiplier * min(self.min_thresh[0], self.server.min_score[0][1])
        th1 = self.multiplier * min(self.min_thresh[1], self.server.min_score[1][1])

        mask0 = d0 < (2 * th0)
        mask1 = d1 < (2 * th1)
        good = [
            self.list_clients[i]
            for i in range(len(self.list_clients))
            if mask0[i] and mask1[i]
        ]

        self.server.good_clients = good
        print(f"Good workers after filtration: {self.server.good_clients}", flush=True)

        aggregated_weights = self.server.global_model.state_dict()
        for i in self.server.good_clients:
            for key, weights in self.server.client_gradients[i].items():
                aggregated_weights[key] = aggregated_weights[key] + weights.to(
                    self.server.device
                ) * (1 / len(self.server.good_clients))
        return self.add_noise(aggregated_weights)

    def _stack_client_safeguards(self, phase):
        # Builds a [C × D] tensor where each row is one client’s flattened safeguard vector
        C = len(self.list_clients)
        D = self.server._total_dim
        mat = torch.empty((C, D), device=self.server.device)
        for i, rank in enumerate(self.list_clients):
            flat = torch.cat(
                [
                    self.server.client_safeguards[phase][rank][k].view(-1)
                    for k in self.server._param_names
                ]
            )
            mat[i] = flat
        return mat

    def _find_medians(self, mat):
        # Compute full pairwise distances (C×C), zero the diag, and pick each row’s median
        dists = torch.cdist(mat, mat, p=2)  # [C, C]
        dists.fill_diagonal_(float("inf"))  # so we don’t pick self-distance
        mid = (mat.size(0) - 1) // 2
        vals, idxs = torch.kthvalue(dists, mid + 1, dim=1)
        return vals, idxs

