import torch
from ..ditto.ditto_client import DittoClient


class pFedMeClient(DittoClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)

    def adjust_model(self):
        if self.do_proximity:
            self.model.to(self.device)
            self.server_model.to(self.device)
            factor = self.proximity * self.optimizer.param_groups[0]["lr"]
            with torch.no_grad():
                aggregated_weights = self.server_model.state_dict()
                for key, weights in self.model.state_dict().items():
                    aggregated_weights[key] = (
                        aggregated_weights[key] * (1 - factor)
                        + weights.to(self.device) * factor
                    )
                self.server_model.load_state_dict(aggregated_weights)

    def get_loss_value(self, outputs, targets):
        loss = super().get_loss_value(outputs, targets)
        self.adjust_model()
        return loss