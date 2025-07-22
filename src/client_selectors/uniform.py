import random as rand

from types import MethodType
from .base import BaseSelector


class UniformSelector(BaseSelector):
    def __init__(self, cfg=None):
        super().__init__(cfg)

    def select_clients_to_train(self, subsample_amount):
        # Just a random sample of clients by default
        return rand.sample([_ for _ in range(self.amount_of_clients)], subsample_amount)

    def change_functionality(self, trainer):
        trainer.server.select_clients_to_train = MethodType(
            UniformSelector.select_clients_to_train, trainer.server
        )
        return trainer
