import numpy as np

from types import MethodType
from .base import BaseSelector


class FedCBS(BaseSelector):
    def __init__(self, cfg, lambda_):
        super().__init__(cfg)
        self.lambda_ = lambda_
        assert (
            "ptbxl" not in self.cfg.train_dataset._target_
        ), "Now FedCBS doesn`t support PTB-XL dataset"

    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        if num_clients_subset == self.amount_of_clients:
            if not server_sampling:
                for i in range(self.amount_of_clients):
                    self.selection_counter[i] += 1
            return list(range(self.cfg.federated_params.amount_of_clients))

        selected_clients = []
        remaining_clients = list(range(self.cfg.federated_params.amount_of_clients))

        # beta_m in paper (= m from experiments setup)
        # equal
        # betas[m - 1] in this code
        betas = [m + 1 for m in range(num_clients_subset)]

        for m in range(num_clients_subset):
            if m == 0:
                # Choose first client
                probabilities = np.array(
                    [
                        1.0 / (self.qcid_fn([c]) ** betas[0])
                        + self.lambda_
                        * np.sqrt(
                            3
                            * np.log(self.cur_round + 1)
                            / (2 * (self.selection_counter[c]))
                        )
                        for c in remaining_clients
                    ]
                )
            elif m == 1:
                # Choose second client
                probabilities = np.array(
                    [
                        1.0
                        / (self.qcid_fn(selected_clients + [c]) ** betas[1])
                        / (
                            (1.0 / (self.qcid_fn(selected_clients) ** betas[0]))
                            + self.lambda_
                            * np.sqrt(
                                3
                                * np.log(self.cur_round + 1)
                                / (2 * (self.selection_counter[c]))
                            )
                        )
                        for c in remaining_clients
                    ]
                )
            else:
                # Choose m-th client (2 < m < M)
                probabilities = np.array(
                    [
                        (self.qcid_fn(selected_clients) ** betas[m - 2])
                        / (
                            (self.qcid_fn(selected_clients + [c]) ** betas[m - 1])
                            if (self.qcid_fn(selected_clients + [c]) ** betas[m - 1])
                            != 0.0
                            else 1e-6
                        )
                        for c in remaining_clients
                    ]
                )
                probabilities = np.array(
                    [
                        (
                            self.qcid_fn(selected_clients)
                            / self.qcid_fn(selected_clients + [c])
                        )
                        ** betas[m - 2]
                        / self.qcid_fn(selected_clients + [c])
                        for c in remaining_clients
                    ]
                )
            probabilities = np.where(
                np.isnan(probabilities) | np.isinf(probabilities), 1e-6, probabilities
            )
            probabilities /= probabilities.sum()  # Normalize probs
            probabilities[probabilities == 0] = 1e-6  # Zeros prob -> small prob

            selected_client = np.random.choice(
                remaining_clients, p=probabilities
            ).item()

            selected_clients.append(selected_client)
            remaining_clients.remove(selected_client)

            if not server_sampling:
                self.selection_counter[selected_client] += 1

        return selected_clients

    def setup_strategy(self, trainer):
        self.df = trainer.df
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        # Naturally we need to set selection counters to zero
        # but if we did it, formulas don`t work
        # then set it to one
        self.selection_counter = np.ones(self.amount_of_clients)

        # For easy code we will imagine that
        # we have all the information about the distribution of clients
        # in paper we need to transmit this information encrypted
        (
            self.client_distr,
            self.client_data_count,
            self.unique_classes,
            self.amount_classes,
        ) = self.get_clients_data_info()

        # Create S from paper, we called it a qcid-matrix
        self.qcid_mtr = self.create_qcid_matrix()

    def change_functionality(self, trainer):
        trainer.server.qcid_mtr = self.qcid_mtr
        trainer.server.selection_counter = self.selection_counter
        trainer.server.client_data_count = self.client_data_count
        trainer.server.amount_classes = self.amount_classes
        trainer.server.lambda_ = self.lambda_

        # Change methods
        trainer.server.select_clients_to_train = MethodType(
            FedCBS.select_clients_to_train, trainer.server
        )
        trainer.server.qcid_fn = MethodType(FedCBS.qcid_fn, trainer.server)

        return trainer

    def qcid_fn(self, clients):
        cur_clients_qcid_mtr = self.qcid_mtr[np.ix_(clients, clients)]
        total_data_count = sum(self.client_data_count[client] for client in clients)

        return np.sum(cur_clients_qcid_mtr) / (total_data_count) ** 2 - (
            1.0 / self.amount_classes
        )

    def get_clients_data_info(self):
        client_distr = {}
        client_data_count = {}
        unique_classes = sorted(self.df.data["target"].unique())
        amount_classes = len(unique_classes)

        for num_client in range(self.amount_of_clients):
            client_data = self.df.data[self.df.data["client"] == num_client]
            target_counts = (
                client_data["target"]
                .value_counts(normalize=True)
                .reindex(unique_classes, fill_value=0)
                .to_dict()
            )
            client_distr[num_client] = target_counts
            client_data_count[num_client] = len(client_data)

        return client_distr, client_data_count, unique_classes, amount_classes

    def create_qcid_matrix(self):
        qcid_mtr = np.zeros((self.amount_of_clients, self.amount_of_clients))

        for i in range(self.amount_of_clients):
            for j in range(self.amount_of_clients):
                vector_i = np.array(list(self.client_distr[i].values()))
                vector_j = np.array(list(self.client_distr[j].values()))
                qcid_mtr[i, j] = (
                    np.dot(vector_i, vector_j)
                    * self.client_data_count[i]
                    * self.client_data_count[j]
                )

        print(f"QCID matrix (S in paper) created. Shape is {qcid_mtr.shape}")
        return qcid_mtr
