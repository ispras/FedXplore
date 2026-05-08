import numpy as np
import ast

from utils.data_utils import print_df_distribution


class HospitalDistribution:
    def __init__(self, min_sample_number, max_sample_number, verbose):
        self.min_sample_number = min_sample_number
        self.max_sample_number = max_sample_number
        self.verbose = verbose

    def split_to_clients(self, df, amount_of_clients, random_state, pathology_names):
        """
        split df by hospitals clients with at least min_sample_number and no more than max_sample_number samples

        :param df: train dataframe
        :param min_sample_number: min number of samples from one client
        :param max_sample_number: max number of samples from one client
        :param pathology_names: list of pathology names
        :param amount_of_clients: amount of participating clients
        """
        if "ID_CLINIC" in df.columns:
            df["client"] = df["ID_CLINIC"]

        train_value_counts = df["client"].value_counts()

        # filter by min_sample_number
        available_clients = train_value_counts[
            train_value_counts
            >= self.min_sample_number & (train_value_counts <= self.max_sample_number)
        ].index.tolist()
        # filter by pathology_names
        client_list = []
        num_clients = amount_of_clients
        for id_clinic in available_clients:
            clinic_id_df = df[df["client"] == id_clinic]
            df_pathology_mask = clinic_id_df["scp_codes"].apply(
                lambda x: [item in ast.literal_eval(x) for item in pathology_names]
            )
            np_pathology_mask = np.array(df_pathology_mask.values.tolist())
            # check that for all path in pathology_names exists at least one positive sample
            entry_to_df = all(
                [any(np_pathology_mask[:, i]) for i in range(len(pathology_names))]
            )
            if entry_to_df:
                client_list.append(id_clinic)
                num_clients -= 1
            if num_clients == 0:
                break

        client_list = np.sort(np.array(client_list))[:amount_of_clients]
        assert (
            len(client_list) == amount_of_clients
        ), f"Number of all clients: {train_value_counts}; "
        "Number clients after min-max samples filtering: {len(client_list)}; "
        "Amount of Clients: {amount_of_clients}. Try to relax [min_sample_number, max_sample_number] range"

        client_to_idx = {client: idx for idx, client in enumerate(client_list.tolist())}
        df["client"] = df["client"].map(client_to_idx)

        if self.verbose:
            print_df_distribution(
                df, len(pathology_names), amount_of_clients, pathology_names
            )
        return df
