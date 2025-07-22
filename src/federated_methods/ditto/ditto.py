from ..personalized.fedavg import PerFedAvg
from ..ditto.ditto_client import DittoClient
from ..ditto.ditto_server import DittoServer


class Ditto(PerFedAvg):
    def __init__(self, strategy, cluster_params, ckpt_path, server_test, proximity):
        super().__init__(strategy, cluster_params, ckpt_path, server_test)
        self.proximity = proximity

    def _init_server(self, cfg):
        self.server = DittoServer(cfg, self.server_test)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = DittoClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.proximity])

    def get_communication_content(self, rank):
        res_dict = super().get_communication_content(rank)
        res_dict["local_model"] = (
            {k: v.cpu() for k, v in self.server.local_models[rank].items()}
            if self.cur_round != 0
            else self.server.global_model.state_dict()
        )
        return res_dict
