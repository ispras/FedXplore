import copy

from ..personalized.client import PerClient


class FedAMPClient(PerClient):
    def __init__(
        self,
        *client_args,
        **client_kwargs,
    ):
        base_client_args = client_args[:2]  # `cfg` and `df` and `models_for_clients`
        super().__init__(*base_client_args, **client_kwargs)
        self.proximity = client_args[2]
        self.client_args = client_args
        self.relative_model = copy.deepcopy(self.model)

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["relative_model"] = (
            lambda state_dict: self.relative_model.load_state_dict(
                {k: v.to(self.device) for k, v in state_dict.items()}
            )
        )
        return pipe_commands_map

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        proximity = (
            0.5
            # * (1 / self.optimizer[0].param_groups[0]['lr']) # in original paper, but even if lr = 1e-3
            * self.proximity
            * sum(
                [
                    (p.float() - q.float().detach()).norm() ** 2
                    for (_, p), (_, q) in zip(
                        self.model.named_parameters(),
                        self.relative_model.named_parameters(),
                    )
                ]
            )
        )
        loss += proximity
        return loss

    def get_grad(self):
        self.model.eval()
        for key, weights in self.model.state_dict().items():
            self.grad[key] = weights.to("cpu")
