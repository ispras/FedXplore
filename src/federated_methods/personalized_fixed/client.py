from ..fedavg.fedavg_client import FedAvgClient


class PersonalizedFixedClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.personalized_model_state = None
        self.personalized_result = None

    def clone_model_state(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def set_personalized_result(self, metrics, loss):
        self.personalized_model_state = self.clone_model_state()
        self.personalized_result = (metrics, float(loss), len(self.valid_dataset))
        self.server_metrics = metrics
        self.server_val_loss = float(loss)

    def get_communication_content(self):
        result = super().get_communication_content()
        result["personalized_model"] = self.personalized_model_state
        result["personalized_metrics"] = self.personalized_result
        return result
