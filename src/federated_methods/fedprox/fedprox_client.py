from ..fedavg.fedavg_client import FedAvgClient


class FedProxClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        base_client_args = client_args[:2]  # `cfg` and `df`
        super().__init__(*base_client_args, **client_kwargs)
        self.client_args = client_args
        self.fed_prox_lambda = self.client_args[2]
        self.num_fedavg_rounds = self.client_args[3]
        self.cur_com_round = None
        self.server_model_state = None

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["current_round"] = self.set_cur_round
        return pipe_commands_map

    def set_cur_round(self, round):
        self.cur_com_round = round

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        if self.cur_com_round > self.num_fedavg_rounds - 1:
            proximity = (
                0.5
                * self.fed_prox_lambda
                * sum(
                    [
                        (p.float() - q.float()).norm() ** 2
                        for (_, p), (_, q) in zip(
                            self.model.state_dict().items(),
                            self.server_model_state.items(),
                        )
                    ]
                )
            )
            loss += proximity
        return loss
