from hydra.utils import instantiate

from ..base.base import Base
from .byzantine_server import ByzantineBaseServer


class ByzantineBase(Base):
    """Abstract class for Byzantine based methods.
    It overrides `aggregate` method:
    1. Performs pre-aggregation, if specified;
    2. Calculate trust scores for clients (default uniform);
    3. Modifies client updates for further processing.

    x^{t+1} = x^t + Σ_i⋹S(w_i*Δ_i^{t}) => Δ_i^{t} --> Δ_i^{t} * w_i * |S| =>
    = > x^{t+1} = x^t + 1/|S| * Σ_i^N (Δ_i^{t})
    """

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        self.preaggregator = None
        if cfg.get("preaggregator") is not None:
            self.preaggregator = instantiate(cfg.preaggregator, server=self.server)

    def _init_server(self, cfg):
        self.trust_df = None
        if cfg.get("trust_dataset"):
            self.trust_df = instantiate(
                cfg.trust_dataset, cfg=cfg, mode="trust", _recursive_=False
            )
        self.server = ByzantineBaseServer(cfg, self.trust_df)

    def count_trust_scores(self):
        # By default each client has equal trust_scores
        return [1 / self.num_clients_subset for _ in range(self.num_clients_subset)]

    def _modify_gradients(self, trust_scores):
        for i, rank in enumerate(self.list_clients):
            print(f"Client {rank} trust score: {trust_scores[i]}")
            modified_client_model_weights = {
                k: v * trust_scores[i] * self.num_clients_subset
                for k, v in self.server.client_gradients[rank].items()
            }
            self.server.client_gradients[rank] = modified_client_model_weights

    def make_pre_aggregation(self):
        if self.preaggregator is not None:
            updated_gradients = self.preaggregator.pre_aggregate(
                [self.server.client_gradients[rank] for rank in self.list_clients]
            )
            for i, rank in enumerate(self.list_clients):
                self.server.client_gradients[rank] = updated_gradients[i]

    def aggregate(self):
        self.make_pre_aggregation()
        trust_scores = self.count_trust_scores()
        self._modify_gradients(trust_scores)
        return super().aggregate()
