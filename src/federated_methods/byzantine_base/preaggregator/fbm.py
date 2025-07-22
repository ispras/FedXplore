import torch

from .base_preaggregator import BasePreaggregator


class FBM(BasePreaggregator):
    def __init__(self, num_byzantines, server):
        super().__init__(server)
        self.num_byzantines = num_byzantines

    def find_nearest_neighbours(self, rank):
        clients_distance = {}
        client_flat = torch.cat(
            [x.flatten() for k, x in self.client_grads[rank].items() if k != "ipm_eps"]
        )
        for neighbour_rank in range(self.num_clients):
            if neighbour_rank == rank:
                continue
            neighbour_flat = torch.cat(
                [
                    x.flatten()
                    for k, x in self.client_grads[neighbour_rank].items()
                    if k != "ipm_eps"
                ]
            )
            clients_distance[neighbour_rank] = torch.linalg.norm(
                neighbour_flat - client_flat, dtype=torch.float32
            ).item()

        return [
            k for k, _ in sorted(clients_distance.items(), key=lambda item: item[1])
        ]

    def nearest_neighbour_mixing(self):
        new_client_gradients = []
        for rank in range(self.num_clients):
            new_client_grad = {}
            sorted_ranks = self.find_nearest_neighbours(rank)
            print(
                f"For client {rank} the list of nearest clients: {sorted_ranks[:self.num_clients - self.num_byzantines]}"
            )
            for key, weights in self.client_grads[rank].items():
                if key == "ipm_eps":
                    continue
                for i in range(self.num_clients - self.num_byzantines):
                    if key not in new_client_grad:
                        new_client_grad[key] = self.client_grads[sorted_ranks[i]][
                            key
                        ] / (self.num_clients - self.num_byzantines)
                    else:
                        new_client_grad[key] = new_client_grad[key] + self.client_grads[
                            sorted_ranks[i]
                        ][key] / (self.num_clients - self.num_byzantines)

            new_client_gradients.append(new_client_grad)

        return new_client_gradients

    def pre_aggregate(self, client_gradients):
        self.client_grads = client_gradients
        self.num_clients = len(self.client_grads)
        assert (
            self.num_clients > self.num_byzantines
        ), f"number of assumed byzantines={self.num_byzantines} should be less than cfg.federated_params.client_subset_size={self.num_clients}"
        return self.nearest_neighbour_mixing()
