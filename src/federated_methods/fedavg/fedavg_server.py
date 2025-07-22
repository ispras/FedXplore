from ..base.base_server import BaseServer
import random as rand

class FedAvgServer(BaseServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        self.amount_classes = None 
        self.cur_round = None
        
    def select_clients_to_train(self, subsample_amount, server_sampling=False):
        # Just a random sample of clients
        return rand.sample(
            [_ for _ in range(self.amount_of_clients)], subsample_amount
        )
