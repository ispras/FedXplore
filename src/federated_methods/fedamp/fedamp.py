import copy
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector
from sklearn.metrics.pairwise import cosine_similarity

from ..personalized.fedavg import PerFedAvg
from ..fedamp.fedamp_client import FedAMPClient


class FedAMP(PerFedAvg):
    def __init__(
        self,
        strategy,
        cluster_params,
        ckpt_path,
        server_test,
        proximity,
        scaling,
        self_value,
    ):
        super().__init__(strategy, cluster_params, ckpt_path, server_test)
        self.proximity = proximity
        self.scaling = scaling
        self.self_value = self_value

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedAMPClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.proximity])

    def define_aggregation_weights(self):
        with torch.no_grad():
            clients_params_vectors = []
            for client_rank in range(self.cfg.federated_params.amount_of_clients):
                personalized_model = self.server.client_gradients[client_rank]
                parameter_list = [
                    param.flatten() for param in personalized_model.values()
                ]
                vec = np.concatenate(parameter_list)
                clients_params_vectors.append(vec)
            clients_params_vectors = np.vstack(clients_params_vectors)
            sim_matrix = cosine_similarity(
                clients_params_vectors, clients_params_vectors
            )
            sim_matrix = sim_matrix / self.scaling
            aggregation_weights = self.softmax(sim_matrix)
            print(f"\nAggregation weights shape {aggregation_weights.shape}\n")
            print(aggregation_weights)
            return aggregation_weights

    def aggregate(self):
        aggr_weights = self.define_aggregation_weights()
        global_model_state_dict = self.server.global_model.state_dict()
        n_clients = self.cfg.federated_params.amount_of_clients

        # Pre-calculation of weighted relative models for each client
        with torch.no_grad():
            weighted_gradients = {rank: {} for rank in range(n_clients)}
            for key in global_model_state_dict.keys():
                grad_stack = torch.stack(
                    [self.server.client_gradients[i][key] for i in range(n_clients)]
                )
                for client_rank in range(n_clients):
                    weights = torch.tensor(
                        aggr_weights[client_rank],
                        dtype=grad_stack.dtype,
                        device=self.server.device,
                    ).view(-1, *([1] * (grad_stack.dim() - 1)))

                    weighted_gradients[client_rank][key] = torch.sum(
                        weights.to("cpu") * grad_stack, dim=0
                    )

        # Init relative_models
        self.server.relative_models = [
            copy.deepcopy(global_model_state_dict) for _ in range(n_clients)
        ]

        # Update relative_models
        for client_rank in range(n_clients):
            for key in global_model_state_dict.keys():
                self.server.relative_models[client_rank][key] = weighted_gradients[
                    client_rank
                ][key]

        return global_model_state_dict

    def get_communication_content(self, rank):
        # Don`t support finetuning by default
        content = {
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
            "strategy": self.clients_strategy[rank],
            # Send the same client model back, expect first round
            "update_model": {
                k: v.cpu()
                for k, v in (
                    self.server.client_gradients[rank].items()
                    if self.cur_round != 0
                    else self.server.global_model.state_dict().items()
                )
            },
            # Also send an aggregated model relative to other clients, expect first round
            "relative_model": {
                k: v.cpu()
                for k, v in (
                    self.server.relative_models[rank].items()
                    if self.cur_round != 0
                    else self.server.global_model.state_dict().items()
                )
            },
        }

        return content

    def softmax(self, x, axis=1):
        if self.self_value is None:
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        else:
            # Weighted softmax with selfvalue
            x = x.copy()
            np.fill_diagonal(x, -np.inf)
            max_x = np.max(
                np.where(np.isfinite(x), x, -np.inf), axis=axis, keepdims=True
            )
            stabilized_x = x - max_x
            exp_x = np.exp(stabilized_x)
            sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True) + 1e-12
            weights = exp_x / sum_exp_x
            weights *= 1 - self.self_value
            np.fill_diagonal(weights, self.self_value)
            return weights
