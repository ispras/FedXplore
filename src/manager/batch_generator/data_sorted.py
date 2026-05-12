from .base import Base


class DataSortedBatchGenerator(Base):
    def __init__(self, batch_size, amount_of_clients, df, *args, **kwargs):
        super().__init__(batch_size, amount_of_clients)
        self.sorted_clients_idx = self.get_sorted_clients_idx(df)

    def get_sorted_clients_idx(self, df):
        """
        Gets sorted clients idx list by data distribution

        Parameters:
            df (pd.DataFrame): The data distribution per client dataframe

        Returns:
            list: The sorted list of clients idx
        """
        counts = df.data["client"].value_counts().index
        return counts.to_list()

    def create_batches(self, current_round_clients):
        set_clients = set(current_round_clients)

        sorted_round_clients = [x for x in self.sorted_clients_idx if x in set_clients]

        self.batches = [
            sorted_round_clients[i : i + self.batch_size]
            for i in range(0, len(sorted_round_clients), self.batch_size)
        ]
        self.num_batches = len(self.batches)
