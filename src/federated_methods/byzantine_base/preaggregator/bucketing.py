import copy
import math
import numpy as np

from .base_preaggregator import BasePreaggregator


class Bucketing(BasePreaggregator):
    def __init__(self, beta, s, server):
        """
        BYZANTINE-ROBUST LEARNING ON HETEROGENEOUS DATASETS VIA BUCKETING
        https://openreview.net/pdf?id=jXKKDEi5vJt

        Args:
            beta (float): momentum coefficient (see Alg. 2)
            s (int): number of elements in one bucket (see Alg. 1)
        """
        super().__init__(server)
        self.beta = beta
        self.s = s
        self.num_clients = self.server.cfg.federated_params.amount_of_clients
        self.active_num_clients = self.server.cfg.federated_params.client_subset_size
        self.client_momentums = [{} for _ in range(self.num_clients)]
        self.buckets = [{} for _ in range(math.ceil(self.active_num_clients / s))]

    def update_client_momentums(self):
        for i, rank in enumerate(self.list_clients):
            for key, _ in self.server.global_model.named_parameters():
                if self.client_momentums[rank].get(key) is None:
                    self.client_momentums[rank][key] = self.client_grads[i][key]
                else:
                    self.client_momentums[rank][key] = (
                        1 - self.beta
                    ) * self.client_grads[i][key] + self.beta * self.client_momentums[
                        rank
                    ][
                        key
                    ]
            for key, _ in self.server.global_model.named_buffers():
                self.client_momentums[rank][key] = self.client_grads[i][key]

    def aragg(self):
        "Algorithm 1"
        np.random.seed(self.server.cfg.random_state + self.server.cur_round)

        permutations = np.random.permutation(self.list_clients)
        bucket_to_clients = {i: [] for i in range(len(self.buckets))}
        for i in range(len(self.buckets)):
            bucket = {k: 0 for k in self.client_momentums[self.list_clients[0]].keys()}
            for j in range(i * self.s, min(len(permutations), (i + 1) * self.s)):
                rank = permutations[j]
                bucket_to_clients[i].append(rank)
                for key, weights in self.client_momentums[rank].items():
                    bucket[key] += weights * (1 / self.s)
            self.buckets[i] = bucket
        # to integrate pre-aggregation into the pipeline
        # we need to be sure that len(updated_gradients) == len(self.client_grads)
        # So, we unroll buckets to clients according to `permutation`
        updated_grads = [{} for _ in range(self.active_num_clients)]
        for i, rank in enumerate(self.list_clients):
            bucket_num = [k for k, v in bucket_to_clients.items() if rank in v]
            assert (
                len(bucket_num) == 1
            ), f"Bucket splitting not full: {bucket_to_clients}"
            bucket_num = bucket_num[0]
            updated_grads[i] = {
                key: self.buckets[bucket_num][key]
                for key, _ in self.server.global_model.state_dict().items()
            }
        return updated_grads

    def pre_aggregate(self, client_gradients):
        self.list_clients = self.server.list_clients
        self.client_grads = client_gradients
        self.update_client_momentums()
        return self.aragg()
