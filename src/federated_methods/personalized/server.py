from ..fedavg.fedavg_server import FedAvgServer
import pandas as pd
import numpy as np


class PerServer(FedAvgServer):
    def __init__(self, cfg, server_test):
        super().__init__(cfg)
        self.server_test = server_test

    def test_global_model(self, dataset="test", require_metrics=True):
        # if any(not df.empty for df, _, _ in self.server_metrics):
        if any(not df.empty for df in self.server_metrics):
            # Print metrics for global model over clusters
            server_cluster_metrics = self.get_cluster_subset(self.server_metrics)
            server_cluster_losses = self.get_cluster_subset(self.server_losses)
            self.print_meaned_metrics(server_cluster_metrics, server_cluster_losses)
            # Print metrics for local models over clusters
            if self.cfg.federated_params.print_client_metrics:
                client_cluster_metrics = self.get_cluster_subset(self.clients_metrics)
                client_cluster_losses = self.get_cluster_subset(self.clients_losses)
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

    def get_cluster_subset(self, values):
        cluster_values = {
            strategy: [values[i] for i in self.strategy_map[strategy]]
            for strategy in self.strategy_map.keys()
        }
        return cluster_values

    def print_meaned_metrics(self, cluster_metrics, cluster_losses):
        for strategy in self.strategy_map.keys():
            print(f"\n-------- Mean {strategy} cluster metrics --------")
            metrics = pd.concat(cluster_metrics[strategy]).groupby(level=0).mean()
            loss = np.mean(cluster_losses[strategy])
            print(f"\nServer Valid Results:\n{metrics}")
            print(f"Server Valid Loss: {loss}")
