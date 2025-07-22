from ..personalized.fedavg import PerFedAvg
from ..fedrep.fedrep_client import FedRepClient


class FedRep(PerFedAvg):
    def __init__(
        self,
        strategy,
        cluster_params,
        ckpt_path,
        server_test,
        warmup_rounds,
    ):
        super().__init__(strategy, cluster_params, ckpt_path, server_test)
        self.warmup_rounds = warmup_rounds

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        # Insert warmup flag in the first place in content
        content_warmup = {"warmup": self.cur_round < self.warmup_rounds}
        new_content = {**content_warmup, **content}

        return new_content

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedRepClient
        self.client_kwargs["client_cls"] = self.client_cls
