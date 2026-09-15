import torch

from ..personalized.method import PersonalizedMethod
from .pfedme_client import pFedMeClient


class pFedMe(PersonalizedMethod):
    def __init__(self, proximity, momentum, k_steps, personal_learning_rate):
        super().__init__()
        self.proximity = proximity
        self.momentum = momentum
        self.k_steps = k_steps
        self.personal_learning_rate = personal_learning_rate

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = pFedMeClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend(
            [self.proximity, self.k_steps, self.personal_learning_rate]
        )

    def aggregate(self):
        aggregated = self.server.global_model.state_dict()
        for key, value in aggregated.items():
            if value.is_floating_point():
                mean_delta = torch.stack(
                    [
                        self.server.client_gradients[rank][key]
                        for rank in self.list_clients
                    ]
                ).mean(dim=0)
                aggregated[key] = value + self.momentum * mean_delta.to(
                    self.server.device
                )
        self.finalize_personalized_round()
        return aggregated
