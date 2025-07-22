import torch
from ..ditto.ditto import Ditto
from ..ditto.ditto_server import DittoServer
from ..pfedme.pfedme_client import pFedMeClient


class pFedMe(Ditto):
    def __init__(
        self,
        strategy,
        cluster_params,
        ckpt_path,
        server_test,
        proximity,
        momentum,
    ):
        super().__init__(strategy, cluster_params, ckpt_path, server_test, proximity)
        self.momentum = momentum

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = pFedMeClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend([self.proximity])

    def _init_server(self, cfg):
        self.server = DittoServer(cfg, self.server_test)

    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()
        num_clients = len(self.list_clients)
        
        for key in aggregated_weights:
            if not aggregated_weights[key].is_floating_point():
                continue
                
            delta_sum = torch.zeros_like(aggregated_weights[key], dtype=torch.float32, device=self.server.device)
            
            for rank in self.list_clients:
                client_delta = self.server.client_gradients[rank][key].to(self.server.device)
                delta_sum += client_delta.float() / num_clients
            
            # Apply momentum: w_k = w_{k-1} + β * avg_delta (equal paper)
            aggregated_weights[key] += (self.momentum * delta_sum).to(aggregated_weights[key].dtype)
        
        return aggregated_weights
