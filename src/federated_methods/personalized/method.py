from ..fedavg.fedavg import FedAvg
from .server import PersonalizedServer


class PersonalizedMethod(FedAvg):
    def _init_server(self, cfg):
        self.server = PersonalizedServer(cfg)

    def _init_client_cls(self):
        assert (
            self.num_clients_subset == self.amount_of_clients
        ), f"{type(self).__name__} requires full client participation"
        super()._init_client_cls()

    def get_personalized_state(self, rank):
        state = self.server.personalized_models[rank]
        return state or {
            key: value.detach().cpu().clone()
            for key, value in self.server.global_model.state_dict().items()
        }

    def finalize_personalized_round(self):
        self.server.evaluate_personalized_models()
        self.server.save_best_personalized_model(self.cur_round)

    def aggregate(self):
        aggregated = super().aggregate()
        self.finalize_personalized_round()
        return aggregated

    def log_evaluation_metrics(self):
        self.logger.log_scalar(
            self.server.personalized_test_loss,
            "test/loss",
            self.cur_round,
        )
        self.logger.log_pandas(
            self.server.personalized_test_metrics,
            "test/",
            self.cur_round,
        )
        self.logger.log_scalar(
            self.server.personalized_validation_loss,
            "val/loss",
            self.cur_round,
        )
        self.logger.log_pandas(
            self.server.personalized_validation_metrics,
            "val/",
            self.cur_round,
        )
