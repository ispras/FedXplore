import numpy as np
from utils.data_utils import print_df_distribution


class DirichletDistribution:
    def __init__(self, alpha, verbose):
        self.alpha = alpha
        self.verbose = verbose

    def split_to_clients(self, df, amount_of_clients, random_state):
        n_classes = df["target"].nunique()
        # Class distribution calculating
        class_distrubution = np.array(
            df["target"].value_counts().sort_index().to_list()
        )

        data_distr = self.dirichlet_distrubution(
            len(df),
            n_classes,
            amount_of_clients,
            class_distrubution,
            random_state,
        )

        # Drop 'client' column
        df["client"] = -1

        client_target_count = {
            i: {j: count for j, count in enumerate(row)}
            for i, row in enumerate(data_distr)
        }

        # Fill 'client' column
        for index, row in df.iterrows():
            target = row["target"]
            for client, counts in client_target_count.items():
                if counts[target] > 0:
                    df.at[index, "client"] = client
                    client_target_count[client][target] -= 1
                    break

        if self.verbose:
            print_df_distribution(df, df["target"].nunique(), amount_of_clients)

        # Check the results
        result = df.groupby(["client", "target"]).size().unstack(fill_value=0)
        for client, row in result.iterrows():
            if client != -1:
                expected = data_distr[client]
                actual = row.tolist()
                assert np.all(
                    expected == actual
                ), f"Mismatch for client {client}: Expected {expected}, Actual {actual}"

        print("\nChecking: All clients have the correct distribution of targets.\n")

        return df

    def dirichlet_distrubution(
        self, total_data_points, num_classes, num_clients, class_distrubution, seed
    ):
        np.random.seed(seed)
        dirichet = np.random.dirichlet(self.alpha * np.ones(num_clients), num_classes)
        data_distr = (dirichet * class_distrubution[:, np.newaxis]).astype(int)
        data_distr = data_distr.transpose()

        total_assigned = data_distr.sum()
        remaining_data_points = total_data_points - total_assigned
        max_per_class = class_distrubution

        class_counts = {i: data_distr[:, i].sum() for i in range(num_classes)}

        # Distribute remaining data (because we use .astype(int))
        if remaining_data_points > 0:
            for i in range(remaining_data_points):
                for class_idx in range(num_classes):
                    if class_counts[class_idx] < max_per_class[class_idx]:
                        client_idx = np.argmin(data_distr.sum(axis=1))
                        data_distr[client_idx, class_idx] += 1
                        class_counts[class_idx] += 1
                        break

        return data_distr
