from ..base.base_server import BaseServer
import pandas as pd
import numpy as np
import os
import torch
from utils.metrics_utils import stopping_criterion
from hydra.utils import instantiate
from utils.utils import create_model_info


class PerServer(BaseServer):
    def __init__(self, cfg, server_test):
        super().__init__(cfg)
        self.server_metrics = [
            (pd.DataFrame(), 0, 0)
            for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.client_metrics = [
            (0, pd.DataFrame()) for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.server_test = server_test

    def test_global_model(self, dataset="test", require_metrics=True):
        if any(not df.empty for df, _, _ in self.server_metrics):
            # Print metrics for global model over clusters
            server_cluster_metrics, server_cluster_losses = self.get_cluster_metrics(
                self.server_metrics
            )
            self.print_meaned_metrics(server_cluster_metrics, server_cluster_losses)
            # Print metrics for local models over clusters
            if self.cfg.federated_params.print_client_metrics:
                client_cluster_losses, client_cluster_metrics = (
                    self.get_cluster_metrics(self.client_metrics)
                )
                print("\n-------- MEANED METRICS AFTER FINETUNING --------")
                self.print_meaned_metrics(client_cluster_metrics, client_cluster_losses)
            # Print metrics for global model on test dataset
            if self.server_test:
                self.last_metrics, self.test_loss = self.model_trainer.test_fn(self)
                print(f"\nServer Test Results:")
                print(self.last_metrics)
                print(f"Server Test Loss: {self.test_loss}")
        else:
            self.global_model.to(self.device)

    def set_client_result(self, client_result):
        # Put client information in accordance with his rank
        self.client_gradients[client_result["rank"]] = client_result["grad"]
        self.server_metrics[client_result["rank"]] = client_result["server_metrics"]
        if self.cfg.federated_params.print_client_metrics:
            self.client_metrics[client_result["rank"]] = client_result["client_metrics"]

    def get_cluster_metrics(self, type_metrics):
        server_metrics = [metrics[0] for metrics in type_metrics]
        val_losses = [metrics[1] for metrics in type_metrics]
        cluster_metrics = {
            strategy: [server_metrics[i] for i in self.strategy_map[strategy]]
            for strategy in self.strategy_map.keys()
        }
        cluster_losses = {
            strategy: [val_losses[i] for i in self.strategy_map[strategy]]
            for strategy in self.strategy_map.keys()
        }
        return cluster_metrics, cluster_losses

    def print_meaned_metrics(self, cluster_metrics, cluster_losses):
        for strategy in self.strategy_map.keys():
            print(f"\n-------- Mean {strategy} cluster metrics --------")
            metrics = pd.concat(cluster_metrics[strategy]).groupby(level=0).mean()
            loss = np.mean(cluster_losses[strategy])
            print(f"\nServer Valid Results:\n{metrics}")
            print(f"Server Valid Loss: {loss}")
