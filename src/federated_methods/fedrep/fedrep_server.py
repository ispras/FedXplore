from collections import OrderedDict

from ..personalized.server import PersonalizedServer


class FedRepServer(PersonalizedServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.local_heads = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        self.local_heads[client_result["rank"]] = client_result["local_head"]
