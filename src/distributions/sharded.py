import random
import numpy as np
import pandas as pd

from utils.data_utils import print_df_distribution


class ShardedDistribution:
    def __init__(
        self,
        n_clusters: int,
        dominant_ratio: float = 0.8,
        num_dominants: int = 1,
        num_connected_clusters: int = -1,
        verbose: bool = True,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if dominant_ratio <= 0 or dominant_ratio >= 1:
            raise ValueError("dominant_ratio must be between 0 and 1 (exclusive).")
        if num_dominants < 1:
            raise ValueError("num_dominants must be at least 1.")

        self.n_clusters = n_clusters
        self.dominant_ratio = dominant_ratio
        self.num_dominants = num_dominants
        self.num_connected_clusters = num_connected_clusters
        self.verbose = verbose

    def split_to_clients(
        self,
        df: pd.DataFrame,
        amount_of_clients: int,
        random_state: int,
    ) -> tuple[pd.DataFrame, list[tuple[int, int, list[int]]]]:
        if amount_of_clients <= 0:
            raise ValueError("amount_of_clients must be a positive integer.")

        num_classes = df["target"].nunique()
        assert (
            self.n_clusters * self.num_dominants == num_classes
        ), f"In sharded distribution major classes must overlap the whole classes. "
        "Found {self.n_clusters} clusters and {self.num_dominants} dominants, "
        "but total classes are {num_classes}, which is not equal to n_clusters * num_dominants."

        df = df.copy().reset_index(drop=True)

        np.random.seed(random_state)
        random.seed(random_state)

        classes = sorted(df["target"].unique())
        n_classes = len(classes)
        if n_classes == 0:
            raise ValueError("Dataframe must contain at least one class in 'target'.")
        if self.num_dominants >= n_classes:
            raise ValueError(
                "num_dominants must be less than the number of unique classes."
            )

        print("Start Sharded Distribution of data")

        class_to_idx = {c: i for i, c in enumerate(classes)}
        counts_per_class = df["target"].value_counts().sort_index().values
        total_points = len(df)

        sizes = self._calculate_client_sizes(total_points, amount_of_clients)
        cluster_of_client = self._assign_clusters_to_clients(amount_of_clients)
        cluster_sizes = self._calculate_cluster_sizes(sizes, cluster_of_client)

        cluster_quota, all_dominants, connected = self._compute_cluster_quotas(
            n_classes,
            cluster_sizes,
            counts_per_class,
            random_state,
        )

        distr = self._distribute_to_clients(
            cluster_quota,
            sizes,
            cluster_of_client,
            cluster_sizes,
            n_classes,
            random_state,
        )

        total_assigned = distr.sum()
        if total_assigned != total_points:
            print(
                f"Warning: Total assigned {total_assigned} != {total_points}. Dropping {total_points - total_assigned} samples to match."
            )
            drop_indices = np.random.choice(
                len(df), total_points - total_assigned, replace=False
            )
            df = df.drop(drop_indices).reset_index(drop=True)

        df = self._assign_clients_to_df(df, distr, class_to_idx, random_state)
        # personalization savings
        self.info_list = self._build_info_list(
            amount_of_clients, cluster_of_client, distr
        )
        self.connected = connected

        if self.verbose:
            self._print_clusterization_info(
                df, amount_of_clients, all_dominants, connected
            )

        return df

    def _distribute_proportionally(self, ideals: np.ndarray, total: int) -> np.ndarray:
        fvals = ideals
        sum_f = fvals.sum()
        if sum_f == 0:
            if total == 0:
                return np.zeros_like(fvals, dtype=int)
            scaled_ideals = np.full(len(fvals), total / len(fvals))
        else:
            scaled_ideals = fvals * (total / sum_f)

        floors = np.floor(scaled_ideals).astype(int)
        fracs = scaled_ideals - floors
        remaining = total - floors.sum()

        additions = np.zeros_like(floors)
        if remaining > 0:
            order = np.argsort(-fracs)
            additions[order[:remaining]] = 1

        return floors + additions

    def _calculate_client_sizes(
        self, total_points: int, amount_of_clients: int
    ) -> np.ndarray:
        base_size = total_points // amount_of_clients
        sizes = np.full(amount_of_clients, base_size, dtype=int)
        remainder = total_points - base_size * amount_of_clients
        sizes[:remainder] += 1
        return sizes

    def _assign_clusters_to_clients(self, amount_of_clients: int) -> np.ndarray:
        clients_per_cluster_base = amount_of_clients // self.n_clusters
        extra = amount_of_clients - clients_per_cluster_base * self.n_clusters
        cluster_of_client = []
        for cl in range(self.n_clusters):
            count = clients_per_cluster_base + (1 if cl < extra else 0)
            cluster_of_client.extend([cl] * count)
        return np.array(cluster_of_client)

    def _calculate_cluster_sizes(
        self, sizes: np.ndarray, cluster_of_client: np.ndarray
    ) -> np.ndarray:
        cluster_sizes = np.zeros(self.n_clusters, dtype=int)
        for cl in range(self.n_clusters):
            cluster_sizes[cl] = sizes[cluster_of_client == cl].sum()
        return cluster_sizes

    def _compute_cluster_quotas(
        self,
        n_classes: int,
        cluster_sizes: np.ndarray,
        counts_per_class: np.ndarray,
        random_state: int,
    ) -> tuple[np.ndarray, list[list[int]], list[set[int]]]:
        np.random.seed(random_state)
        random.seed(random_state)

        all_dominants = [
            [
                (cl * self.num_dominants + i) % n_classes
                for i in range(self.num_dominants)
            ]
            for cl in range(self.n_clusters)
        ]

        connected = [set() for _ in range(self.n_clusters)]
        if self.num_connected_clusters > 0:
            group_size = self.num_connected_clusters + 1
            clusters_list = list(range(self.n_clusters))
            random.shuffle(clusters_list)
            groups = [
                clusters_list[i : i + group_size]
                for i in range(0, self.n_clusters, group_size)
            ]

            for group in groups:
                for j in range(len(group)):
                    for k in range(j + 1, len(group)):
                        cl1 = group[j]
                        cl2 = group[k]
                        connected[cl1].add(cl2)
                        connected[cl2].add(cl1)

        cluster_quota_f = np.zeros((self.n_clusters, n_classes), dtype=float)
        for cl in range(self.n_clusters):
            dominants = all_dominants[cl]
            if self.num_connected_clusters <= 0:
                other_dominants_lists = all_dominants[:cl] + all_dominants[cl + 1 :]
            else:
                connected_cls = list(connected[cl])
                other_dominants_lists = [all_dominants[c] for c in connected_cls]

            non_dominants_set = set()
            for other in other_dominants_lists:
                non_dominants_set.update(other)
            non_dominants = list(non_dominants_set)

            if len(non_dominants) == 0:
                non_dominants = list(set(range(n_classes)) - set(dominants))

            ratios = np.zeros(n_classes)
            if dominants:
                ratios[dominants] = self.dominant_ratio / len(dominants)
            if non_dominants:
                ratios[non_dominants] = (1 - self.dominant_ratio) / len(non_dominants)

            cluster_quota_f[cl] = ratios * cluster_sizes[cl]

        cluster_quota = np.zeros_like(cluster_quota_f, dtype=int)
        for cls in range(n_classes):
            fvals = cluster_quota_f[:, cls]
            total_needed = counts_per_class[cls]
            cluster_quota[:, cls] = self._distribute_proportionally(fvals, total_needed)

        if not np.allclose(cluster_quota.sum(axis=0), counts_per_class):
            print(
                "Warning: Quota sums do not match global counts exactly. Adjusting quotas."
            )
            for cls in range(n_classes):
                diff = counts_per_class[cls] - cluster_quota[:, cls].sum()
                if diff != 0:
                    home_cl = (cls // self.num_dominants) % self.n_clusters
                    if diff > 0:
                        cluster_quota[home_cl, cls] += diff
                    elif diff < 0:
                        if cluster_quota[home_cl, cls] >= -diff:
                            cluster_quota[home_cl, cls] += diff
                        else:
                            non_zero = np.where(cluster_quota[:, cls] > 0)[0]
                            for nc in non_zero:
                                take = min(cluster_quota[nc, cls], -diff)
                                cluster_quota[nc, cls] -= take
                                diff += take
                                if diff == 0:
                                    break

        return cluster_quota, all_dominants, connected

    def _distribute_to_clients(
        self,
        cluster_quota: np.ndarray,
        sizes: np.ndarray,
        cluster_of_client: np.ndarray,
        cluster_sizes: np.ndarray,
        n_classes: int,
        random_state: int,
    ) -> np.ndarray:
        np.random.seed(random_state)
        distr = np.zeros((len(sizes), n_classes), dtype=int)
        global_assigned = np.zeros(n_classes, dtype=int)

        for cl in range(self.n_clusters):
            client_indices = np.where(cluster_of_client == cl)[0]
            client_sizes = sizes[client_indices]
            cluster_size = cluster_sizes[cl]
            if cluster_size == 0:
                continue

            dominants = [
                (cl * self.num_dominants + i) % n_classes
                for i in range(self.num_dominants)
            ]

            for dom in dominants:
                dom_quota = cluster_quota[cl, dom]
                if dom_quota == 0:
                    continue
                ideals = dom_quota * client_sizes / cluster_size
                assigned_dom = self._distribute_proportionally(ideals, dom_quota)
                distr[client_indices, dom] += assigned_dom
                global_assigned[dom] += assigned_dom.sum()

            cq_remain = cluster_quota[cl].copy()
            for dom in dominants:
                cq_remain[dom] -= distr[client_indices, dom].sum()

            remaining_capacities = client_sizes - distr[client_indices].sum(axis=1)
            client_order = np.argsort(-remaining_capacities)
            for ci_idx in client_order:
                client = client_indices[ci_idx]
                capacity = sizes[client] - distr[client].sum()
                if capacity <= 0:
                    continue

                candidates = np.where(cq_remain > 0)[0]
                if len(candidates) == 0:
                    break

                cand_vals = global_assigned[candidates]
                sort_order = np.argsort(cand_vals)
                sorted_cands = candidates[sort_order]
                cand_quotas = cq_remain[sorted_cands]
                sum_cq = cand_quotas.sum()
                if sum_cq == 0:
                    break

                ideals = capacity * (cand_quotas / sum_cq)
                assignments = self._distribute_proportionally(ideals, capacity)

                for j, sc in enumerate(sorted_cands):
                    assignments[j] = min(assignments[j], cq_remain[sc])

                distr[client, sorted_cands] += assignments
                cq_remain[sorted_cands] -= assignments
                global_assigned[sorted_cands] += assignments

            if cq_remain.sum() > 0:
                self._handle_unassigned_quotas(
                    distr, cq_remain, sizes, client_indices, global_assigned
                )

        self._fix_client_totals(distr, sizes)

        return distr

    def _handle_unassigned_quotas(
        self,
        distr: np.ndarray,
        cq_remain: np.ndarray,
        sizes: np.ndarray,
        client_indices: np.ndarray,
        global_assigned: np.ndarray,
    ) -> None:
        rem_classes = np.where(cq_remain > 0)[0]
        for cls in rem_classes:
            need = cq_remain[cls]
            rem_caps = sizes[client_indices] - distr[client_indices].sum(axis=1)
            if rem_caps.sum() > 0:
                ideals = need * rem_caps / rem_caps.sum()
                additions = self._distribute_proportionally(ideals, need)
                distr[client_indices, cls] += additions
                global_assigned[cls] += additions.sum()
                cq_remain[cls] = 0
            else:
                global_rem_caps = sizes - distr.sum(axis=1)
                if global_rem_caps.sum() > 0:
                    ideals = need * global_rem_caps / global_rem_caps.sum()
                    additions = self._distribute_proportionally(ideals, need)
                    distr[:, cls] += additions
                    global_assigned[cls] += additions.sum()
                    cq_remain[cls] = 0
                else:
                    for _ in range(need):
                        donor = np.argmax(distr.sum(axis=1) - sizes)
                        if distr[donor].sum() <= sizes[donor]:
                            raise ValueError(
                                "Cannot place remaining tokens: no capacity."
                            )
                        cls_with_extra = np.where(distr[donor] > 0)[0]
                        if len(cls_with_extra) == 0:
                            break
                        take_cls = cls_with_extra[0]
                        distr[donor, take_cls] -= 1
                        distr[donor, cls] += 1
                        global_assigned[cls] += 1
                    if need > 0:
                        raise ValueError("Couldn't place all remaining tokens.")

    def _fix_client_totals(self, distr: np.ndarray, sizes: np.ndarray) -> None:
        client_totals = distr.sum(axis=1)
        diffs = sizes - client_totals
        need_clients = np.where(diffs > 0)[0]
        extra_clients = np.where(diffs < 0)[0]

        extra_diffs = client_totals[extra_clients] - sizes[extra_clients]
        extra_order = np.argsort(-extra_diffs)
        extra_clients = extra_clients[extra_order]

        for rc in need_clients:
            need = diffs[rc]
            for ec in extra_clients:
                if need == 0:
                    break
                cls_with_extra = np.where(distr[ec] > 0)[0]
                cls_order = np.argsort(-distr[ec, cls_with_extra])
                sorted_cls = cls_with_extra[cls_order]
                for cls in sorted_cls:
                    if need == 0:
                        break
                    take = min(distr[ec, cls], need)
                    distr[ec, cls] -= take
                    distr[rc, cls] += take
                    need -= take

        if not np.array_equal(distr.sum(axis=1), sizes):
            raise ValueError("Failed to fix client totals.")

    def _assign_clients_to_df(
        self,
        df: pd.DataFrame,
        distr: np.ndarray,
        class_to_idx: dict,
        random_state: int,
    ) -> pd.DataFrame:
        np.random.seed(random_state)
        amount_of_clients, n_classes = distr.shape
        df["client"] = -1
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        rem_counts = [
            {cls: distr[i, cls] for cls in range(n_classes)}
            for i in range(amount_of_clients)
        ]

        for idx, row in df.iterrows():
            cls = class_to_idx[row["target"]]
            perm = np.random.permutation(amount_of_clients)
            assigned = False
            for client in perm:
                if rem_counts[client][cls] > 0:
                    df.at[idx, "client"] = int(client)
                    rem_counts[client][cls] -= 1
                    assigned = True
                    break
            if not assigned:
                for client in perm:
                    if sum(rem_counts[client].values()) > 0:
                        df.at[idx, "client"] = int(client)
                        rem_counts[client][cls] = max(0, rem_counts[client][cls] - 1)
                        assigned = True
                        break
            if not assigned:
                raise ValueError("Failed to assign sample to client.")
        return df

    def _build_info_list(
        self,
        amount_of_clients: int,
        cluster_of_client: np.ndarray,
        distr: np.ndarray,
    ) -> list[tuple[int, int, list[int]]]:
        return [
            (i, int(cluster_of_client[i]), distr[i].tolist())
            for i in range(amount_of_clients)
        ]

    def _print_clusterization_info(
        self, df, amount_of_clients, all_dominants, connected
    ) -> None:
        print(
            "Clusterization strategy is sequential, i.e. "
            f"Cluster {0} has dominant classes {all_dominants[0]}, etc."
        )
        if self.num_connected_clusters > 0:
            print("\nCluster Connections:")
            for cl in range(self.n_clusters):
                print(f"Cluster {cl} connected to: {sorted(list(connected[cl]))}")

        print_df_distribution(df, df["target"].nunique(), amount_of_clients)
