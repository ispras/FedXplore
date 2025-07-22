from ..personalized.server import PerServer


class DittoServer(PerServer):
    def __init__(self, cfg, server_test):
        super().__init__(cfg, server_test)
        self.local_models = [
            None for i in range(self.cfg.federated_params.amount_of_clients)
        ]

    def set_client_result(self, client_result):
        # Put client information in accordance with his rank
        super().set_client_result(client_result)
        self.local_models[client_result["rank"]] = client_result["local_model"]
