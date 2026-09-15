from collections import OrderedDict

from ..fedavg.fedavg_server import FedAvgServer


class PersonalizationToyServer(FedAvgServer):
    def __init__(self, cfg):
        super().__init__(cfg)
        count = cfg.federated_params.amount_of_clients
        self.personalized_models = [OrderedDict() for _ in range(count)]
        self.personalized_metrics = [None for _ in range(count)]
        self.personalized_losses = [0.0 for _ in range(count)]

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        rank = client_result["rank"]
        metrics, loss, _ = client_result["personalized_metrics"]
        self.personalized_models[rank] = client_result["personalized_model"]
        self.personalized_metrics[rank] = metrics
        self.personalized_losses[rank] = loss
