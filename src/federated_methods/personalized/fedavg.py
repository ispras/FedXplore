import torch

from ..fedavg.fedavg import FedAvg
from .client import PerClient
from .server import PerServer
from .strategy import *


class PerFedAvg(FedAvg):
    def __init__(self, strategy, ckpt_path, server_test):
        self.strategy = strategy
        self.ckpt_path = ckpt_path
        self.server_test = server_test
        super().__init__()

    def _init_client_cls(self):
        assert (
            self.cfg.federated_params.client_subset_size
            == self.cfg.federated_params.amount_of_clients
        ), """We currently support a personalization scenario 
            where all available clients are used in each round"""

        super()._init_client_cls()
        self.client_cls = PerClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.define_clusters()

    def _init_server(self, cfg):
        self.server = PerServer(cfg, self.server_test)

    def define_clusters(self):
        self.num_clients = self.cfg.federated_params.amount_of_clients
        match self.strategy:
            case "sharded":
                self.strategy = ShardedStrategy()

            case "base":
                self.strategy = BaseStrategy()

            case "filter":
                self.strategy = FilterStrategy()

            case _:
                raise ValueError(f"No such cluster split type {self.strategy}")

        self.strategy_map, self.clients_strategy = self.strategy.split_clients(self)
        self.server.strategy_map = self.strategy_map
        print(f"Cluster mapping: {self.strategy_map}")

    def load_checkpoint(self):
        weights = torch.load(self.ckpt_path, map_location=self.server.device)["model"]
        self.server.global_model.load_state_dict(weights)

    def get_communication_content(self, rank):
        # Fine-tune option
        if self.cur_round == 0 and self.ckpt_path is not None:
            self.load_checkpoint()

        # In we need additionaly send client cluster strategy
        content = super().get_communication_content(rank)
        content["strategy"] = self.strategy.get_client_payload(rank)
        return content
