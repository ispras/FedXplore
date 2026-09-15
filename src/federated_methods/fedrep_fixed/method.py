import copy

from ..personalized_fixed.method import PersonalizedFixedMethod
from .client import FedRepFixedClient
from .server import FedRepFixedServer


class FedRepFixed(PersonalizedFixedMethod):
    def __init__(self, head_local_epochs, representation_local_epochs):
        super().__init__()
        self.head_local_epochs = head_local_epochs
        self.representation_local_epochs = representation_local_epochs

    def _init_server(self, cfg):
        self.server = FedRepFixedServer(cfg)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = FedRepFixedClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.extend(
            [self.head_local_epochs, self.representation_local_epochs]
        )

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        state = copy.deepcopy(self.server.global_model.state_dict())
        state.update(self.server.local_heads[rank])
        content["client_state"] = state
        content.pop("update_model")
        return content

    def aggregate(self):
        weights = self.calculate_aggregation_weights()
        aggregated = copy.deepcopy(self.server.global_model.state_dict())
        for key in aggregated:
            if key.startswith("representation."):
                aggregated[key] = aggregated[key] + sum(
                    self.server.client_gradients[rank][key] * weights[index]
                    for index, rank in enumerate(self.list_clients)
                )
        self.evaluate_aggregated_representation(aggregated)
        return aggregated

    def evaluate_aggregated_representation(self, aggregated):
        validation_metrics = []
        validation_losses = []
        test_metrics = []
        test_losses = []
        for clients_batch in self.manager.create_batches(self.list_clients):
            self.manager.set_ranks_to_procs(clients_batch)
            for pipe_num, rank in enumerate(clients_batch):
                state = copy.deepcopy(aggregated)
                state.update(self.server.local_heads[rank])
                self.server.send_content_to_client(
                    pipe_num,
                    {
                        "attack_type": (
                            self.client_map_round[rank],
                            self.attack_configs[self.client_map_round[rank]],
                        ),
                        "evaluation_state": state,
                    },
                )
            for pipe_num, rank in enumerate(clients_batch):
                result = self.server.rcv_content_from_client(pipe_num)
                self.server.personalized_models[rank] = result["personalized_model"]
                validation_metrics.append(result["validation_metrics"])
                validation_losses.append(result["validation_loss"])
                test_metrics.append(result["test_metrics"])
                test_losses.append(result["test_loss"])
        self.print_diagnostic_metrics(
            validation_metrics, validation_losses, test_metrics, test_losses
        )
