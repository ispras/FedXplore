from ..byzantine_base.byzantine import ByzantineBase
from .zeno_server import ZenoServer


class Zeno(ByzantineBase):
    def __init__(self, ro, b, use_buffers):
        super().__init__()
        self.ro = ro
        self.b = b
        self.use_buffers = use_buffers
        self.initial_global_model_state = None

    def _init_server(self, cfg):
        super()._init_server(cfg)
        self.server = ZenoServer(cfg, self.trust_df, self.use_buffers, self.ro, self.b)
        self.amount_clients = int(self.num_clients_subset * self.b)

    def aggregate(self):
        self.make_pre_aggregation()
        round_sds = self.server.find_highest_sds()
        aggregated_weights = self.server.global_model.state_dict()
        for i, rank in enumerate(self.list_clients):
            if round_sds[i]:
                for key, weights in self.server.client_gradients[rank].items():
                    aggregated_weights[key] = aggregated_weights[key] + weights.to(
                        self.server.device
                    ) * (1 / self.amount_clients)
        return aggregated_weights
