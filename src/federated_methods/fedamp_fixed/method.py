import torch
import torch.nn.functional as functional

from ..personalized_fixed.method import PersonalizedFixedMethod
from .client import FedAMPFixedClient
from .server import FedAMPFixedServer


class FedAMPFixed(PersonalizedFixedMethod):
    def __init__(self, proximity, scaling, self_value):
        super().__init__()
        self.proximity = proximity
        self.scaling = scaling
        self.self_value = self_value

    def _init_server(self, cfg):
        self.server = FedAMPFixedServer(cfg)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedAMPFixedClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.append(self.proximity)

    def get_communication_content(self, rank):
        state = self.get_personalized_state(rank)
        relative = self.server.relative_models[rank] or state
        return {
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
            "client_model": state,
            "relative_model": relative,
        }

    def attention_weights(self):
        vectors = [
            torch.cat([value.flatten() for value in state.values()]).float()
            for state in self.server.personalized_models
        ]
        normalized = functional.normalize(torch.stack(vectors), dim=1)
        similarities = normalized @ normalized.T / self.scaling
        similarities.fill_diagonal_(-float("inf"))
        weights = torch.softmax(similarities, dim=1) * (1 - self.self_value)
        weights.fill_diagonal_(self.self_value)
        return weights

    def aggregate(self):
        weights = self.attention_weights()
        states = self.server.personalized_models
        for rank in range(self.amount_of_clients):
            self.server.relative_models[rank] = {
                key: torch.stack(
                    [states[other][key] for other in range(self.amount_of_clients)]
                )
                .mul(
                    weights[rank].view(
                        -1,
                        *([1] * states[0][key].dim()),
                    )
                )
                .sum(dim=0)
                for key in states[rank]
            }
        self.print_personalized_diagnostics()
        return self.server.global_model.state_dict()
