from ..fedavg.fedavg_client import FedAvgClient
from .strategy import STRATEGY_REGISTRY, BaseStrategy


class PerClient(FedAvgClient):
    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["strategy"] = self.map_client_strategy
        return pipe_commands_map

    def map_client_strategy(self, strategy_payload):
        assert isinstance(
            strategy_payload, dict
        ), f"Unsupported strategy payload type: {type(strategy_payload)}"

        strategy_key = strategy_payload.get("strategy_key")
        if strategy_key is None:
            raise ValueError("Client payload must include a strategy_key.")

        strategy_cls = STRATEGY_REGISTRY.get(strategy_key)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown strategy key received: {strategy_key}. If you add a new strategy, please register it in STRATEGY_REGISTRY."
            )

        init_kwargs = strategy_payload.get("init_kwargs") or {}
        strategy_instance = strategy_cls(**init_kwargs)
        strategy_instance.apply_client_payload(self, strategy_payload)
        self.strategy_payload = strategy_payload
