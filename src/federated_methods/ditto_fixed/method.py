from ..personalized_fixed.method import PersonalizedFixedMethod
from .client import DittoFixedClient


class DittoFixed(PersonalizedFixedMethod):
    def __init__(self, proximity):
        super().__init__()
        self.proximity = proximity

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = DittoFixedClient
        self.client_kwargs["client_cls"] = self.client_cls
        self.client_args.append(self.proximity)

    def get_communication_content(self, rank):
        content = super().get_communication_content(rank)
        content["personalized_model"] = self.get_personalized_state(rank)
        return content

    def aggregate(self):
        aggregated_weights = super().aggregate()
        self.print_personalized_diagnostics()
        return aggregated_weights
