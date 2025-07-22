import numpy as np

from types import MethodType
from .base import BaseSelector


class Pow(BaseSelector):
    def __init__(self, cfg, candidate_set_size):
        super().__init__(cfg)
        self.candidate_set_size = candidate_set_size

    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        if num_clients_subset == self.amount_of_clients:
            return list(range(self.cfg.federated_params.amount_of_clients))

        # Randomly select a subset of clients from the total pool
        # based on their dataset size
        # (a.k.a. A in paper, |A| = d)
        candidate_clients_list = np.random.choice(
            range(self.amount_of_clients),
            size=self.candidate_set_size,
            replace=False,
            p=self.clients_probs,
        )
        candidate_clients_list = candidate_clients_list.tolist()

        if not server_sampling:
            print(f"Current clients losses: {self.clients_losses}", flush=True)
            print(f"Current candidate clients: {candidate_clients_list}", flush=True)

        # Sort the selected candidates by their loss values
        candidate_clients_list.sort(
            key=lambda client_rank: self.clients_losses[client_rank], reverse=True
        )

        if not server_sampling:
            print(
                f"Sorted candidate clients: {sorted(candidate_clients_list)}",
                flush=True,
            )
            for idx, cl in enumerate(candidate_clients_list):
                print(f"{idx} : Client{cl} : {self.clients_losses[cl]}", flush=True)

        # Select the top `amount_of_clients` clients from the sorted list
        selected_clients = candidate_clients_list[:num_clients_subset]

        if not server_sampling:
            print(f"Selected clients: {selected_clients}", flush=True)

        return selected_clients

    def change_functionality(self, trainer):
        trainer.server.candidate_set_size = self.candidate_set_size
        trainer.server.clients_losses = [
            np.inf for _ in range(trainer.server.amount_of_clients)
        ]

        # Change methods

        # Rework parse content function
        trainer.orig_parse_communication_content = MethodType(
            getattr(type(trainer), "parse_communication_content"),
            trainer,
        )
        trainer.parse_communication_content = MethodType(
            Pow.parse_communication_content,
            trainer,
        )

        # Rework get content function in client side
        trainer.client_cls.orig_get_communication_content = (
            trainer.client_cls.get_communication_content
        )
        trainer.client_cls.get_communication_content = Pow.get_communication_content

        # Setup initial probabilities
        trainer.server.set_probs_to_choise_client = MethodType(
            Pow.set_probs_to_choise_client, trainer.server
        )
        trainer.server.df = trainer.df
        trainer.server.clients_probs = trainer.server.set_probs_to_choise_client()

        # Change client selection function
        trainer.server.select_clients_to_train = MethodType(
            Pow.select_clients_to_train, trainer.server
        )
        return trainer

    def set_probs_to_choise_client(self):
        clients_df_len = [
            len(self.df.data[self.df.data["client"] == i])
            for i in range(self.amount_of_clients)
        ]
        clients_probs = [
            client_len / sum(clients_df_len) for client_len in clients_df_len
        ]
        print(f"Probabilities of client selection: {clients_probs}")

        return clients_probs

    def parse_communication_content(self, client_result):
        self.orig_parse_communication_content(client_result)
        if client_result["rank"] in self.list_clients:
            self.server.clients_losses[client_result["rank"]] = client_result[
                "client_loss"
            ]

    def get_communication_content(self):
        content = self.orig_get_communication_content()
        self.client_val_loss, self.client_metrics = self.model_trainer.client_eval_fn(
            self
        )
        content["client_loss"] = self.client_val_loss

        return content
