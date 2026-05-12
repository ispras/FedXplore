from ..uniform_fedavg.uniform_fedavg import UniformFedAvg
from .fedprox_client import FedProxClient


class FedProx(UniformFedAvg):
    def __init__(self, fed_prox_lambda, num_fedavg_rounds):
        super().__init__()
        self.fed_prox_lambda = fed_prox_lambda
        self.num_fedavg_rounds = num_fedavg_rounds

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedProxClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.fed_prox_lambda, self.num_fedavg_rounds])

    def get_communication_content(self, rank):
        # In fedprox we need additionaly send current round to warmup
        content = super().get_communication_content(rank)
        content["current_round"] = self.cur_round
        return content
