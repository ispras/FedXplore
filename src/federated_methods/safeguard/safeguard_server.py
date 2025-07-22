from collections import OrderedDict

from ..base.base_server import BaseServer
from hydra.utils import instantiate


class Safeguard_server(BaseServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.good_clients = list(range(cfg.federated_params.amount_of_clients))
        self.client_safeguards = [
            [OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)],
            [OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)],
        ]
        self.min_score = [0, 0]
        self.grad_med_acum = [OrderedDict(), OrderedDict()]
