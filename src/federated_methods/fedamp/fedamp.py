import torch
import torch.nn.functional as functional

from ..personalized.method import PersonalizedMethod
from .fedamp_client import FedAMPClient
from .fedamp_server import FedAMPServer


class FedAMP(PersonalizedMethod):
    def __init__(self, proximity, scaling, self_value):
        super().__init__()
        self.proximity = proximity
        self.scaling = scaling
        self.self_value = self_value

    def _init_server(self, cfg):
        self.server = FedAMPServer(cfg)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedAMPClient
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
            torch.cat(
                [
                    value.flatten()
                    for value in state.values()
                    if value.is_floating_point()
                ]
            ).float()
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
            relative = {}
            for key, value in states[rank].items():
                if value.is_floating_point():
                    stacked = torch.stack(
                        [state[key] for state in states]
                    )
                    relative[key] = stacked.mul(
                        weights[rank].view(-1, *([1] * value.dim()))
                    ).sum(dim=0)
                else:
                    relative[key] = value.detach().cpu().clone()
            self.server.relative_models[rank] = relative
        self.finalize_personalized_round()
        return self.server.global_model.state_dict()
