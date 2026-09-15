from collections import OrderedDict

from ..personalized_fixed.server import PersonalizedFixedServer


class FedAMPFixedServer(PersonalizedFixedServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.relative_models = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]
