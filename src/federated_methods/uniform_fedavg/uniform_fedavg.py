import numpy as np

from ..fedavg.fedavg import FedAvg


class UniformFedAvg(FedAvg):
    def calculate_aggregation_weights(self):
        # Uniform averaging
        return [1 / self.num_clients_subset] * self.num_clients_subset
