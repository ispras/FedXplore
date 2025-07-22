import torch
import copy
import torch.nn.functional as F
from collections import OrderedDict

from ..byzantine_base.byzantine_server import ByzantineBaseServer


class RecessServer(ByzantineBaseServer):
    def __init__(self, cfg, baseline_decreased_score, init_trust_score):
        super().__init__(cfg)
        self.num_clients_subset = self.cfg.federated_params.client_subset_size
        self.trust_scores = [init_trust_score for _ in range(self.num_clients_subset)]
        self.prev_client_gradients = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.baseline_decreased_score = baseline_decreased_score
        self.init_trust_score = init_trust_score

    def adjust_model(self, rank: int) -> None:
        new_model_state = OrderedDict()
        for key, weights in self.global_model.state_dict().items():
            new_model_state[key] = (
                weights.float()
                + self.client_gradients[rank][key].to(self.device).float()
            )
        return new_model_state

    def calculate_trust_scores(self) -> None:
        for i, rank in enumerate(self.list_clients):
            abnormality_alpha = self.calculate_abnormality_alpha(rank)
            self.trust_scores[i] = (
                self.init_trust_score
                - abnormality_alpha * self.baseline_decreased_score
            )
        self.trust_scores = F.softmax(torch.tensor(self.trust_scores), dim=0).tolist()

    def calculate_abnormality_alpha(
        self,
        rank: int,
    ) -> int:
        old_grad = torch.cat(
            [x.flatten() for x in self.prev_client_gradients[rank].values()]
        )
        new_grad = torch.cat(
            [x.flatten() for x in self.client_gradients[rank].values()]
        )
        cos_sim = torch.dot(old_grad, new_grad) / (
            torch.linalg.norm(old_grad) * torch.linalg.norm(new_grad)
        )
        return (-cos_sim / torch.linalg.norm(new_grad)).item()

    def gradient_resetting(self) -> None:
        # set clients gradients to zeros
        for rank in self.list_clients:
            for key, weights in self.global_model.state_dict().items():
                self.client_gradients[rank][key] = torch.zeros_like(
                    weights, dtype=torch.float
                )

    def gradient_normalization(self) -> None:
        for rank in self.list_clients:
            grad_flat = torch.cat(
                [x.flatten() for x in self.client_gradients[rank].values()]
            )
            grad_norm = torch.linalg.norm(grad_flat, dtype=torch.float32).item()
            for key, grad in self.client_gradients[rank].items():
                self.client_gradients[rank][key] = grad.float() / grad_norm
            self.prev_client_gradients[rank] = copy.deepcopy(
                self.client_gradients[rank]
            )
