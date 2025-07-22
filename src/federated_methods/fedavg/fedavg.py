from ..base.base import Base
from .fedavg_client import FedAvgClient
from .fedavg_server import FedAvgServer
import time

from utils.attack_utils import (
    set_client_map_round,
)
from hydra.utils import instantiate


class FedAvg(Base):
    def __init__(self, num_clients_subset):
        super().__init__()
        self.num_clients_subset = num_clients_subset

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

        self.amount_of_clients = cfg.federated_params.amount_of_clients
        if self.num_clients_subset > self.amount_of_clients:
            print(
                f"Number of all clients: {self.amount_of_clients} is less that provided number in subset: {self.num_clients_subset}. We use all avaiable clients"
            )
            self.num_clients_subset = self.amount_of_clients

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedAvgClient
        self.client_kwargs["client_cls"] = self.client_cls

    def _init_server(self, cfg):
        self.server = FedAvgServer(cfg)
        self.server.amount_classes = self.df.num_classes

    def calculate_ts(self):
        clients_df_len = [
            len(self.df.data[self.df.data["client"] == i]) for i in self.list_clients
        ]
        return [client_len / sum(clients_df_len) for client_len in clients_df_len]

    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()

        for idx, rank in enumerate(self.list_clients):
            for key, weights in self.server.client_gradients[rank].items():
                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + weights.to(self.server.device) * self.ts[idx]
                )

        return aggregated_weights

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content["need_train"] = rank in self.list_clients
        return content

    def parse_communication_content(self, client_result):
        if client_result["rank"] in self.list_clients:
            super().parse_communication_content(client_result)
        else:
            self.server.server_metrics[client_result["rank"]] = client_result[
                "server_metrics"
            ]

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches
        self.server.global_model = instantiate(
            self.cfg.model, num_classes=self.df.num_classes
        )

        for cur_round in range(self.rounds):
            print(f"\nRound number: {cur_round}")
            begin_round_time = time.time()
            self.cur_round = cur_round
            self.server.cur_round = cur_round

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_attack_map,
                self.attack_rounds,
                self.attack_scheme,
                cur_round,
            )

            self.list_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            self.list_clients.sort()
            print(f"Clients on this communication: {self.list_clients}")
            print(
                f"Amount of clients on this communication: {len(self.list_clients)}\n"
            )

            self.train_round()

            self.server.test_global_model()
            self.server.save_best_model(cur_round)

            self.ts = self.calculate_ts()
            print(f"Client weights for aggregation on this communication {self.ts}")

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        print("Shutdown clients, federated learning end", flush=True)
        self.manager.stop_train()
