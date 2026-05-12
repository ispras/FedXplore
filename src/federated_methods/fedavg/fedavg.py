import os
import time
import copy
import tempfile
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from collections import OrderedDict

from .fedavg_server import FedAvgServer
from .fedavg_client import FedAvgClient

from utils.data_utils import create_distribution_md
from utils.logging_utils import build_client_participation_histogram
from utils.attack_utils import (
    map_attack_clients,
    set_attack_rounds,
    set_client_map_round,
    load_attack_configs,
    apply_synchronized_attack,
)


class FedAvg:
    def __init__(self):
        self.server = None
        self.client = None
        self.rounds = 0
        self.terminated_event = None
        self.client_map_round = None
        self.list_clients = None
        self.aggr_weights = None

    def _init_federated(self, cfg):
        self.logger = instantiate(cfg.logger, run_dir=cfg.single_run_dir)
        # Init train dataset
        self.train_dataset = instantiate(
            cfg.train_dataset, cfg=cfg, mode="train", _recursive_=False
        )
        self.cfg = self.train_dataset.get_cfg()

        self.num_clients_subset = self.cfg.federated_params.client_subset_size
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients

        if self.num_clients_subset > self.amount_of_clients:
            print(
                f"""Number of all clients: {self.amount_of_clients} 
                is less that provided number in subset: 
                {self.num_clients_subset}. We use all avaiable clients"""
            )
            self.num_clients_subset = self.amount_of_clients

        # We go through the data once
        self.clients_df_len = np.array(
            [
                len(self.train_dataset.data[self.train_dataset.data["client"] == i])
                for i in range(self.amount_of_clients)
            ]
        )
        self.client_times = np.zeros((self.amount_of_clients,))
        self.client_selection_records = []

        # Initialize the server, and client's base
        self.attack_setup(self.cfg)
        self._init_server(self.cfg)
        self._init_client_cls()
        self._init_manager()

    def _init_server(self, cfg):
        self.server = FedAvgServer(cfg)

    def _init_client_cls(self):
        self.client_cls = FedAvgClient
        self.client_args = [self.cfg, self.train_dataset]
        self.client_kwargs = {
            "client_cls": self.client_cls,
            "pipe": None,
            "rank": None,
            "attack_type": None,
        }

    def _init_manager(self):
        self.manager = instantiate(
            self.cfg.manager, self.cfg, self.server, self.train_dataset
        )

    def attack_setup(self, cfg):
        self.rounds = cfg.federated_params.communication_rounds
        self.client_attack_map = map_attack_clients(
            cfg.federated_params.clients_attack_types,
            cfg.federated_params.prop_attack_clients,
            cfg.federated_params.amount_of_clients,
        )
        self.attack_scheme = cfg.federated_params.attack_scheme
        self.attack_rounds = set_attack_rounds(
            cfg.federated_params.prop_attack_rounds, self.rounds, self.attack_scheme
        )
        self.attack_configs = load_attack_configs(
            cfg, cfg.federated_params.clients_attack_types
        )

    def calculate_aggregation_weights(self):
        # Weighted averaging proportional to the local client data sizes
        cur_clients_df_len = self.clients_df_len[self.list_clients]
        aggr_weights = (cur_clients_df_len / sum(cur_clients_df_len)).tolist()
        return aggr_weights

    def aggregate(self):
        # Aggragation client weights into global model
        self.aggr_weights = self.calculate_aggregation_weights()

        aggregated_weights = self.server.global_model.state_dict()
        for idx, rank in enumerate(self.list_clients):
            for key, gradients in self.server.client_gradients[rank].items():
                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + gradients.to(self.server.device) * self.aggr_weights[idx]
                )

        return aggregated_weights

    def get_communication_content(self, rank):
        # In fedavg we need to send model after aggregate and
        # attack type for every client
        return {
            "update_model": {
                k: v.cpu() for k, v in self.server.global_model.state_dict().items()
            },
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
        }

    def parse_communication_content(self, client_result):
        # In fedavg we recive result_dict from every client
        self.server.set_client_result(client_result)
        print(f"Client {client_result['rank']} finished in {client_result['time']}")
        self.client_times[client_result["rank"]] = client_result["time"]
        if self.cfg.federated_params.print_client_metrics:
            # client_result['client_metrics'] = (loss, metrics)
            client_loss, client_metrics = (
                client_result["client_metrics"][0],
                client_result["client_metrics"][1],
            )
            print(client_metrics)
            print(f"Validation loss: {client_loss}\n")

    def cleanup(self):
        # Emptying memory forcibly
        self.server.client_gradients = [
            OrderedDict() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]
        self.server.server_metrics = [
            pd.DataFrame() for _ in range(self.cfg.federated_params.amount_of_clients)
        ]

    def train_round(self):
        self.clients_loader = self.manager.create_batches(self.list_clients)

        for clients_batch in self.clients_loader:
            print(f"Current batch of clients is {clients_batch}", flush=True)
            # Manager reinit clients with new ranks
            self.manager.set_ranks_to_procs(clients_batch)

            # Send content to clients to start local learning
            for pipe_num, rank in enumerate(clients_batch):
                content = self.get_communication_content(rank)
                self.server.send_content_to_client(pipe_num, content)

            # Waiting end of local learning and recieve content from clients
            for pipe_num, rank in enumerate(clients_batch):
                content = self.server.rcv_content_from_client(pipe_num)
                self.parse_communication_content(copy.deepcopy(content))

        # apply ipm or alie attack on central node
        self.server.client_gradients = apply_synchronized_attack(
            self.list_clients,
            self.server.client_gradients,
            self.client_map_round,
            self.attack_configs,
            self.server.global_model,
        )

    def log_round(self):
        # Update checkpoint path in logger, because best model save with round number
        self.logger.checkpoint_path = self.server.checkpoint_path

        if self.cur_round == 0:
            # First round. Log info about run, client distribution
            md_distr = create_distribution_md(
                self.train_dataset,
                num_classes=self.cfg.training_params.num_classes,
                num_clients=self.amount_of_clients,
            )

            self.logger.save_artifact(md_distr, "client_distribution.md")
            self.logger.log_run_info(self.cfg)

        # Log test metrics
        self.logger.log_scalar(
            float(self.server.test_loss), "test/loss", self.cur_round
        )
        self.logger.log_pandas(self.server.last_metrics, "test/", self.cur_round)

        # Log val metrics
        self.logger.log_scalar(
            float(self.server.latest_validation_loss), "val/loss", self.cur_round
        )
        self.logger.log_pandas(
            self.server.latest_validation_metrics, "val/", self.cur_round
        )

        if self.cfg.federated_params.print_client_metrics:
            # collect client metrics, if provided
            clients = self.list_clients
            clients_metrics = self.server.clients_metrics
            clients_losses = self.server.clients_losses

            for rank in clients:
                client_loss = clients_losses[rank]
                client_metrics = clients_metrics[rank]
                self.logger.log_scalar(client_loss, f"clients/client_{rank}/loss", self.cur_round)
                self.logger.log_pandas(client_metrics, f"clients/client_{rank}/", self.cur_round)

        # Work with time
        cur_times = self.client_times[self.client_times != 0]
        self.logger.log_scalar(self.round_time, "time/round_time", self.cur_round)
        self.logger.log_scalar(cur_times.max(), "time/max_cl_time", self.cur_round)
        self.logger.log_scalar(cur_times.min(), "time/min_cl_time", self.cur_round)
        self.logger.log_scalar(cur_times.std(), "time/std_cl_time", self.cur_round)
        self.logger.log_scalar(cur_times.mean(), "time/mean_cl_time", self.cur_round)
        self.client_times = np.zeros((self.amount_of_clients,))

        # Work with client selection
        self.client_selection_records.append(
            {"round": self.cur_round, "clients": self.list_clients}
        )
        df = pd.DataFrame.from_records(
            self.client_selection_records,
            columns=["round", "clients"],
        )
        self.logger.save_artifact(
            df.to_csv(index=False),
            "client_selection/selection_log.csv",
        )

        # Plot client selection frequincy bar plot
        with tempfile.TemporaryDirectory() as tmp:
            plot_path = os.path.join(tmp, "participation_histogram.png")

            build_client_participation_histogram(
                selection_df=df,
                num_clients=self.amount_of_clients,
                save_path=plot_path,
            )

            self.logger.save_artifact(
                open(plot_path, "rb").read(),
                "client_selection/participation_histogram.png",
            )

        # Generate conflunce report
        self.logger.generate_confluence_report()

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.server.global_model = instantiate(
            self.cfg.model, num_classes=self.train_dataset.num_classes
        )
        self.server.criterion = self.server.criterion.to(self.server.device)

        for cur_round in range(self.rounds):
            print(f"\nRound number: {cur_round}")
            begin_round_time = time.time()
            self.cur_round = cur_round
            self.server.cur_round = cur_round

            self.server.test_global_model()

            print("\nTraining started\n")

            # Setup attack on current round
            self.client_map_round = set_client_map_round(
                self.client_attack_map,
                self.attack_rounds,
                self.attack_scheme,
                cur_round,
            )

            # Select clients on current round
            self.list_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            self.list_clients.sort()
            self.server.list_clients = self.list_clients
            print(f"Clients on this communication: {self.list_clients}")
            print(
                f"Amount of clients on this communication: {len(self.list_clients)}\n"
            )

            # Client training
            self.train_round()

            self.server.save_best_model(cur_round)

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            self.round_time = time.time() - begin_round_time
            print(f"Round time: {self.round_time}", flush=True)
            self.log_round()
            self.cleanup()

        print("Shutdown clients, federated learning end", flush=True)
        self.logger.end_logging()
        self.manager.stop_train()
