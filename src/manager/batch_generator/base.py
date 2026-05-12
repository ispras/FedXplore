class Base:
    def __init__(self, batch_size, amount_of_clients):
        self.amount_of_clients: int = amount_of_clients
        self.ranks: list = [i for i in range(self.amount_of_clients)]
        self.batch_size: int = self.define_batch_len(batch_size)

        self.current_idx: int = 0
        self.num_batches: int = 0
        self.batches: list = list()

    def create_batches(self, current_round_clients):
        raise NotImplementedError("Base does not have a batch creation function")

    def define_batch_len(self, batch_size):
        return min(self.amount_of_clients, batch_size)
