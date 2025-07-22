import torch
import torch.nn.functional as F

from ..byzantine_base.byzantine import ByzantineBase
from hydra.utils import instantiate
from .simbant_server import SimBANTServer


class SimBANT(ByzantineBase):
    def __init__(self, trust_test_samples, prob_temperature, similarity_type):
        super().__init__()
        self.trust_test_samples = trust_test_samples
        self.prob_temperature = prob_temperature
        self.similarity_type = similarity_type
        assert self.similarity_type in [
            "cosine",
            "cosine_targets",
        ], f"We support only ['cosine', 'cosine_targets'] similarity metrics, you provide: {self.similarity_type}"

    def _init_server(self, cfg):
        super()._init_server(cfg)
        self.server = SimBANTServer(
            cfg, self.trust_df, self.trust_test_samples, self.prob_temperature
        )
        self.num_clients = len(self.server.client_gradients)

    def count_trust_scores(self):
        targets, server_probs, clients_probs = self.server.trust_eval_models()
        trust_scores = self.bant_similarity(targets, server_probs, clients_probs)
        return trust_scores

    def bant_similarity(self, targets, server_probs, clients_probs):
        trust_scores = []
        for i in range(self.num_clients_subset):
            if self.similarity_type == "cosine":
                similarity_score = F.cosine_similarity(
                    clients_probs[i], server_probs, dim=1
                )
                mean = similarity_score.mean()
                std = similarity_score.std()
                client_trust_score = max(mean - std, 0.0001)
            if self.similarity_type == "cosine_targets":
                similarity_score = F.cosine_similarity(clients_probs[i], targets, dim=1)
                client_trust_score = similarity_score.mean()

            trust_scores.append(client_trust_score)

        # normalize trust scores
        if self.similarity_type == "cosine":
            trust_scores = [ts / sum(trust_scores) for ts in trust_scores]
        if self.similarity_type == "cosine_targets":
            trust_scores = F.softmax(torch.stack(trust_scores) / 0.05, dim=0)

        return trust_scores
