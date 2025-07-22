from ..byzantine_base.byzantine import ByzantineBase
from .bant_server import BANTServer
from hydra.utils import instantiate


class BANT(ByzantineBase):
    def __init__(self, momentum_beta):
        super().__init__()
        self.momentum_beta = momentum_beta

    def _init_server(self, cfg):
        super()._init_server(cfg)
        self.server = BANTServer(cfg, self.trust_df)
        self.num_clients = cfg.federated_params.amount_of_clients
        self.prev_trust_scores = [1 / self.num_clients] * self.num_clients

    def count_trust_scores(self):
        self.prev_trust_scores_subset = [
            self.prev_trust_scores[rank] for rank in self.list_clients
        ]
        server_loss, client_losses = self.server.get_trust_losses()
        trust_scores = self.calculate_trust_score(server_loss, client_losses)
        return trust_scores

    def calculate_trust_score(self, server_loss, client_losses):
        # Calculate loss diff
        trust_scores = [max(server_loss - cl, 0) for cl in client_losses]
        # Create a trust scores with momentum
        sum_ts = sum(trust_scores)
        beta = self.momentum_beta if sum_ts else 0.001
        trust_scores = (
            [ts / sum_ts for ts in trust_scores]
            if sum_ts
            else [1 / self.num_clients_subset] * self.num_clients_subset
        )

        momentum_ts = [
            (1 - beta) * prev_ts + beta * cur_ts
            for prev_ts, cur_ts in zip(self.prev_trust_scores_subset, trust_scores)
        ]
        for i, rank in enumerate(self.list_clients):
            self.prev_trust_scores.insert(rank, momentum_ts[i])

        # Make idicating
        trust_scores = [
            prev_ts if cur_ts else cur_ts
            for prev_ts, cur_ts in zip(momentum_ts, trust_scores)
        ]
        # normalize ts
        trust_scores = [ts / sum(trust_scores) for ts in trust_scores]
        return trust_scores
