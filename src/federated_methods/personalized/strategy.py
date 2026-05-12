from abc import ABC
import numpy as np
from omegaconf import open_dict


class BaseStrategy(ABC):
    strategy_key = "base"

    def __init__(self, cluster_params=None):
        self.cluster_distribution = cluster_params
        self.strategy_map = {}
        self.clients_strategy = {}

    def get_strategy_names(self, context=None):
        # Defines strategy names
        return "origin"

    def split_clients(self, context):
        """Split clients into clusters according to the strategy. The BaseStrategy puts all clients into one cluster

        Args: context (PerFedAvg): The federated learning context
        Returns: strategy_map (dict): A mapping from cluster to lists of client ranks
                clients_strategy (dict): A mapping from client ranks to their assigned cluster
        """
        strategy = self.get_strategy_names(context)
        num_clients = context.cfg.federated_params.amount_of_clients
        strategy_map = {strategy: [client for client in range(num_clients)]}
        clients_strategy = {client: strategy for client in range(num_clients)}
        self.set_assignments(strategy_map, clients_strategy)
        return strategy_map, clients_strategy

    def set_assignments(self, strategy_map, clients_strategy):
        self.strategy_map = strategy_map
        self.clients_strategy = clients_strategy

    def get_client_payload(self, client_rank):
        # Forms a content which will be sent to the client
        # init_kwargs is nessesary to reconstruct the strategy instance on the client side
        cluster = self.clients_strategy[client_rank]
        return {
            "strategy_key": self.strategy_key,
            "cluster": cluster,
            "init_kwargs": self.get_init_kwargs(),
        }

    def apply_client_payload(self, client, payload):
        # Applies the payload received from the server to the client
        # In BaseStrategy, we just set the client's strategy attribute
        cluster = payload.get("cluster")
        if cluster is None:
            raise ValueError("Client payload must contain cluster information.")
        client.strategy = cluster

    def get_init_kwargs(self):
        return (
            {"cluster_params": self.cluster_distribution}
            if self.cluster_distribution is not None
            else {}
        )


class ShardedStrategy(BaseStrategy):
    strategy_key = "sharded"

    def get_strategy_names(self, context):
        # Sharded distribution already splits clients into clusters, so we simply enumerate them.
        return [
            f"Cluster_{i}" for i in range(context.train_dataset.distribution.n_clusters)
        ]

    def split_clients(self, context):
        # The ShardedStrategy uses the clustering from the data distribution
        strategies = self.get_strategy_names(context)
        distr_info = context.train_dataset.distribution.info_list

        clients_strategy = {
            client_id: strategies[cluster_id]
            for client_id, cluster_id, distr in distr_info
        }
        strategy_map = {strategy: [] for strategy in strategies}

        for client, cluster in clients_strategy.items():
            strategy_map[cluster].append(client)

        self.set_assignments(strategy_map, clients_strategy)
        return strategy_map, clients_strategy


class FilterStrategy(BaseStrategy):
    strategy_key = "filter"

    def get_strategy_names(self, context=None):
        # we model clusters by client behaviour of preprocessing their data
        return ["strong_filter", "origin"]

    def split_clients(self, context):
        # Ecg Trainer assertion
        fl_trainer = context.cfg.model_trainer._target_.split(".")[-1]
        assert (
            fl_trainer == "EcgTrainer"
        ), f"we use FilterStrategy only for Ecg FL pipeline. You provide: {fl_trainer}"

        # The FilterStrategy randomly assigns half of the clients to use strong filtering
        rng = context.cfg.random_state
        num_clients = context.cfg.federated_params.amount_of_clients
        filter_clients = np.random.default_rng(rng).choice(
            num_clients, size=num_clients // 2, replace=False
        )
        strategy_map = {
            "strong_filter": filter_clients.tolist(),
            "origin": [
                client for client in range(num_clients) if client not in filter_clients
            ],
        }
        clients_strategy = {
            client: ("strong_filter" if client in filter_clients else "origin")
            for client in range(num_clients)
        }
        self.set_assignments(strategy_map, clients_strategy)
        return strategy_map, clients_strategy

    def apply_client_payload(self, client, payload):
        super().apply_client_payload(client, payload)
        if client.strategy == "strong_filter":
            transforms = [
                {
                    "_target_": "ecglib.preprocessing.preprocess.ButterworthFilter",
                    "filter_type": "lowpass",
                    "fs": 500,
                    "n": 4,
                    "Wn": 0.3,
                },
                {
                    "_target_": "ecglib.preprocessing.preprocess.ButterworthFilter",
                    "filter_type": "highpass",
                    "fs": 500,
                    "n": 4,
                    "Wn": 0.001,
                },
                {
                    "_target_": "ecglib.preprocessing.preprocess.IIRNotchFilter",
                    "fs": 500,
                    "w0": 0.1,
                },
                {
                    "_target_": "ecglib.preprocessing.preprocess.ButterworthFilter",
                    "filter_type": "lowpass",
                    "fs": 500,
                    "n": 4,
                    "Wn": 0.05,
                },
            ]

            self.set_augmentation(client, transforms, dataset_type="train")
            self.set_augmentation(client, transforms, dataset_type="valid")

    def set_augmentation(self, client, transforms, dataset_type="train"):
        dataset = (
            client.train_dataset if dataset_type == "train" else client.valid_dataset
        )

        with open_dict(dataset.dataset_cfg.augmentation):
            dataset.dataset_cfg.augmentation.prob = 1.0
            dataset.dataset_cfg.augmentation.transforms = transforms

        augmentation = dataset.get_augmentation()
        client.train_loader.dataset.augmentation = augmentation


STRATEGY_REGISTRY = {
    BaseStrategy.strategy_key: BaseStrategy,
    ShardedStrategy.strategy_key: ShardedStrategy,
    FilterStrategy.strategy_key: FilterStrategy,
}
