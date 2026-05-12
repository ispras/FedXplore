from .base import Base


class SequentialBatchGenerator(Base):
    def __init__(self, batch_size, amount_of_clients, *args, **kwargs):
        super().__init__(batch_size, amount_of_clients)

    def create_batches(self, current_round_clients):
        self.batches = [
            current_round_clients[i : i + self.batch_size]
            for i in range(0, len(current_round_clients), self.batch_size)
        ]
        self.num_batches = len(self.batches)
