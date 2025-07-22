import torch
import numpy as np
from ..base.base import Base
from .client import PerClient
from .server import PerServer
from .strategy import *


class PerFedAvg(Base):
    def __init__(self, strategy, cluster_params, ckpt_path, server_test):
        self.strategy = strategy
        self.cluster_params = np.array(cluster_params)
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
                self.strategy = ShardedStrategy(self.cluster_params)

            case "base":
                self.strategy = BaseStrategy()

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
        content["strategy"] = self.clients_strategy[rank]
        return content

    def parse_communication_content(self, client_result):
        # In fedavg we recive result_dict from every client
        self.server.set_client_result(client_result)
        print(f"Client {client_result['rank']} finished in {client_result['time']}")
        if self.cfg.federated_params.print_client_metrics:
            # client_result['client_metrics'] = (loss, metrics)
            client_loss, client_metrics = (
                client_result["client_metrics"][0],
                client_result["client_metrics"][1],
            )
            print(client_metrics)
            print(f"Validation loss: {client_loss}\n")
