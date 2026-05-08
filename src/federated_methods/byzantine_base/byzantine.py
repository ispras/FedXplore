from hydra.utils import instantiate

from ..uniform_fedavg.uniform_fedavg import UniformFedAvg
from .byzantine_server import ByzantineBaseServer


class ByzantineBase(UniformFedAvg):
    """Abstract class for Byzantine based methods.
    It overrides `aggregate` method:
    1. Performs pre-aggregation, if specified;
    2. Calculate trust scores for clients (default uniform);
    """

    def _init_federated(self, cfg):
        super()._init_federated(cfg)
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

    # def count_trust_scores(self):
    #     # By default each client has equal trust_scores
    #     return [1 / self.num_clients_subset for _ in range(self.num_clients_subset)]

    # def _modify_gradients(self, trust_scores):
    #     for i, rank in enumerate(self.list_clients):
    #         print(f"Client {rank} trust score: {trust_scores[i]}")
    #         modified_client_model_weights = {
    #             k: v * trust_scores[i] * self.num_clients_subset
    #             for k, v in self.server.client_gradients[rank].items()
    #         }
    #         self.server.client_gradients[rank] = modified_client_model_weights

    def make_pre_aggregation(self):
        if self.preaggregator is not None:
            updated_gradients = self.preaggregator.pre_aggregate(
                [self.server.client_gradients[rank] for rank in self.list_clients]
            )
            for i, rank in enumerate(self.list_clients):
                self.server.client_gradients[rank] = updated_gradients[i]

    def aggregate(self):
        self.make_pre_aggregation()
        aggregated_weights = super().aggregate()
        print("\nClient trust scores:")
        for rank, score in enumerate(self.aggr_weights):
            print(f"\tClient {rank}: {score:.4f}")
        print("\n\n")
        return aggregated_weights
