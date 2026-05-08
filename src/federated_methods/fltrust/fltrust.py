import torch
from collections import OrderedDict
from torch.nn.functional import relu

from ..byzantine_base.byzantine import ByzantineBase
from .fltrust_server import FLTrustServer
from hydra.utils import instantiate


class FLTrust(ByzantineBase):
    def __init__(self, use_buffers):
        super().__init__()
        self.use_buffers = use_buffers

    def _init_server(self, cfg):
        super()._init_server(cfg)
        self.server = FLTrustServer(cfg, self.trust_df)

    def calculate_aggregation_weights(self):
        self.server.fltrust_train()
        trust_scores = self.calculate_trust_scores()
        self.normalize_magnitudes()
        return trust_scores

    def calculate_trust_scores(self):
        self.client_directions = []
        needed_state = (
            self.server.server_grad.keys()
            if self.use_buffers
            else [name for name, _ in self.server.global_model.named_parameters()]
        )
        self.server_direction = torch.cat(
            [self.server.server_grad[key].flatten() for key in needed_state]
        )
        trust_scores = []

        for i, rank in enumerate(self.list_clients):
            self.client_directions.append(
                torch.cat(
                    [
                        self.server.client_gradients[rank][key].flatten()
                        for key in needed_state
                    ]
                )
            )
            trust_scores.append(
                self.client_trust_score(
                    self.server_direction, self.client_directions[i]
                )
            )
        # normalize trust scores
        trust_scores = [ts / sum(trust_scores) for ts in trust_scores]
        return trust_scores

    def normalize_magnitudes(self):
        for i, rank in enumerate(self.list_clients):
            normalized_client_gradients = OrderedDict()
            for key, weights in self.server.client_gradients[rank].items():
                normalized_client_gradients[key] = (
                    weights
                    * torch.norm(self.server_direction)
                    / torch.norm(self.client_directions[i])
                )
            self.server.client_gradients[rank] = normalized_client_gradients

    def client_trust_score(self, server_direction, client_direction):
        cosine_similarity = torch.dot(server_direction, client_direction) / (
            torch.norm(server_direction) * torch.norm(client_direction)
        )
        relu_score = relu(cosine_similarity)
        return relu_score
