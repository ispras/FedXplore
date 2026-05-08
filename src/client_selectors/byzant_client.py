import random as rand
from types import MethodType
from .base import BaseSelector
from utils.attack_utils import set_client_map_round


class ByzantClientSelector(BaseSelector):
    def __init__(self, cfg, prob_byzant_round, percent_byzants):
        super().__init__(cfg)
        assert (
            cfg.federated_params.attack_scheme == "constant"
        ), "ByzantClientSelector works only with constant attack_scheme!"
        
        self.prob_byzant_round = prob_byzant_round
        self.percent_byzants = percent_byzants
        print(
            f"Byzant client selector: prob_byzant_round={prob_byzant_round}, percent_byzants={percent_byzants}"
        )

    def select_clients_to_train(self, subsample_amount):
        if rand.random() < self.prob_byzant_round:
            num_byzant = int(subsample_amount * self.percent_byzants)
            num_honest = subsample_amount - num_byzant

            selected_byzant = rand.sample(self.byzant_clients, num_byzant)

            honest_clients = [
                client
                for client in range(self.amount_of_clients)
                if client not in self.byzant_clients
            ]
            selected_honest = rand.sample(honest_clients, num_honest)

            selected_clients = selected_byzant + selected_honest
            print(
                f"Byzant Round. Selected {num_byzant} byzant clients and {num_honest} honest clients"
            )
            return selected_clients
        else:
            return rand.sample(list(range(self.amount_of_clients)), subsample_amount)

    def setup_strategy(self, trainer):
        client_map_round = set_client_map_round(
            trainer.client_attack_map,
            trainer.attack_rounds,
            trainer.attack_scheme,
            cur_round=0,
        )
        self.byzant_clients = [
            rank
            for rank in range(trainer.amount_of_clients)
            if client_map_round[rank] != "no_attack"
        ]

    def change_functionality(self, trainer):
        trainer.server.select_clients_to_train = MethodType(
            ByzantClientSelector.select_clients_to_train, trainer.server
        )
        trainer.server.byzant_clients = self.byzant_clients
        trainer.server.prob_byzant_round = self.prob_byzant_round
        trainer.server.percent_byzants = self.percent_byzants
        print(f"Server has byzant clients:\n{trainer.server.byzant_clients}\n")
        return trainer
