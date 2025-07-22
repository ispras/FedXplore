from abc import ABC
from utils.data_utils import create_sharded_df, print_df_distribution


class BaseStrategy(ABC):
    def __init__(self, cluster_params=None):
        self.cluster_distribution = cluster_params

    def get_strategy_names(self):
        return "origin"

    def split_clients(self, context):
        # Put all client to origin cluster
        strategy = self.get_strategy_names()
        num_clients = context.cfg.federated_params.amount_of_clients
        strategy_map = {strategy: [client for client in range(num_clients)]}
        clients_strategy = {client: strategy for client in range(num_clients)}
        return strategy_map, clients_strategy


class ShardedStrategy(BaseStrategy):
    def __init__(self, cluster_params):
        super().__init__(cluster_params)
        self.n_clusters = int(cluster_params[0])
        self.dominant_ratio = cluster_params[1]

    def get_strategy_names(self, context):
        return [f"shard_{i}" for i in range(self.n_clusters)]

    def split_clients(self, context):
        strategies = self.get_strategy_names(context)
        num_clients = context.cfg.federated_params.amount_of_clients

        context.df.data, distr_info = create_sharded_df(
            context.df.data,
            self.n_clusters,
            context.cfg.federated_params.amount_of_clients,
            dominant_ratio=self.dominant_ratio,
            random_state=context.cfg.random_state,
        )

        # Update the original data in the dataset class for client-side work
        context.df.orig_data = context.df.data
        context.client_args[1] = context.df

        print("\n-------- Distribution after clustering --------\n")
        print_df_distribution(
            context.df.data,
            context.cfg.training_params.num_classes,
            num_clients,
            # context.cfg.task_params.pathology_names,
        )

        clients_strategy = {
            client_id: strategies[cluster_id]
            for client_id, cluster_id, distr in distr_info
        }
        strategy_map = {strategy: [] for strategy in strategies}

        for client, cluster in clients_strategy.items():
            strategy_map[cluster].append(client)

        return strategy_map, clients_strategy
