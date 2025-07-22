from ..base.base_client import BaseClient


class FedAvgClient(BaseClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.need_train = False

    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["need_train"] = self.set_need_train
        return pipe_commands_map

    def set_need_train(self, need_train):
        self.need_train = need_train

    def get_communication_content(self):
        if self.need_train:
            return super().get_communication_content()

        result_dict = {
            "rank": self.rank,
            "server_metrics": (
                self.server_metrics,
                self.server_val_loss,
                len(self.valid_dataset),
            ),
        }
        return result_dict

    def train(self):
        if self.need_train:
            super().train()
        else:
            self.server_val_loss, self.server_metrics = self.model_trainer.client_eval_fn(
                self
            )
