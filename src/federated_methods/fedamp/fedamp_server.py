from collections import OrderedDict

from ..personalized.server import PersonalizedServer


class FedAMPServer(PersonalizedServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.relative_models = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]
