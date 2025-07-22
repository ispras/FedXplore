import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def get_dataset_loader(
    dataset,
    cfg,
    drop_last=True,
):
    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=(dataset.mode == "train"),
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    assert (
        len(loader) > 0
    ), f"len(dataloader) is 0, either lower the batch size, or put drop_last=False"
    return loader


def dirichlet_distrubution(
    total_data_points, num_classes, num_clients, alpha, verbose, seed=42
):
    np.random.seed(seed)
    dirichet = np.random.dirichlet(alpha * np.ones(num_clients), num_classes)
    data_distr = (dirichet * total_data_points / num_classes).astype(int)
    data_distr = data_distr.transpose()

    total_assigned = data_distr.sum()
    remaining_data_points = total_data_points - total_assigned
    max_per_class = total_data_points // num_classes

    class_counts = {i: data_distr[:, i].sum() for i in range(num_classes)}

    # Distribute remaining data (because we use .astype(int))
    if remaining_data_points > 0:
        for i in range(remaining_data_points):
            for class_idx in range(num_classes):
                if class_counts[class_idx] < max_per_class:
                    client_idx = np.argmin(data_distr.sum(axis=1))
                    data_distr[client_idx, class_idx] += 1
                    class_counts[class_idx] += 1
                    break

    if verbose:
        print("Total usage data:", data_distr.sum())
        for i, x in enumerate(data_distr):
            x_str = " ".join(f"{num:>4}" for num in x)
            print(f"Client {i:>2} | {x_str} | {sum(x):>5}")

    return data_distr


def create_dirichlet_df(df, amount_of_clients, alpha, verbose=True, random_state=42):
    n_classes = df["target"].nunique()
    data_distr = dirichlet_distrubution(
        len(df),
        n_classes,
        amount_of_clients,
        alpha,
        verbose,
        random_state,
    )

    # Drop 'client' column
    df["client"] = -1

    client_target_count = {
        i: {j: count for j, count in enumerate(row)} for i, row in enumerate(data_distr)
    }

    # Fill 'client' column
    for index, row in df.iterrows():
        target = row["target"]
        for client, counts in client_target_count.items():
            if counts[target] > 0:
                df.at[index, "client"] = client
                client_target_count[client][target] -= 1
                break

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


def create_dirichlet_multilabel_df(
    df, amount_of_clients, alpha, verbose, min_sample_number, seed, pathology_names
):
    """
    Splits df into C clients with a Dirichlet(alpha)-prescribed share for each label and the neutral class.
    df['target'] is assumed to be a list/array of 0/1 of length K.
    Returns a new DataFrame with a 'client' column.
    """

    # Build label matrix Y (N x K)
    Y = np.vstack(df["target"].values)
    N, K = Y.shape
    C = amount_of_clients

    # Identify positives vs neutrals
    is_neutral = Y.sum(axis=1) == 0
    pos_idx = np.where(~is_neutral)[0]
    neu_idx = np.where(is_neutral)[0]

    # 1) Compute Dirichlet capacities
    # 1a) Positives per label
    label_counts = Y[pos_idx].sum(axis=0).astype(int)
    rng = np.random.RandomState(seed)
    dir_samples = rng.dirichlet([alpha] * C, size=K)
    cap_pos = (dir_samples * label_counts[:, None]).astype(int)
    # Fix rounding per label
    for j in range(K):
        rem = label_counts[j] - cap_pos[j].sum()
        if rem > 0:
            fracs = dir_samples[j] * label_counts[j] - cap_pos[j]
            for c in np.argsort(-fracs)[:rem]:
                cap_pos[j, c] += 1
    cap_pos = cap_pos.T  # shape (C, K)

    # 1b) Neutral class
    neutral_count = len(neu_idx)
    rng2 = np.random.RandomState(seed + 1)
    p_neu = rng2.dirichlet([alpha] * C)
    cap_neu = (p_neu * neutral_count).astype(int)
    rem = neutral_count - cap_neu.sum()
    if rem > 0:
        fracs = p_neu * neutral_count - cap_neu
        for c in np.argsort(-fracs)[:rem]:
            cap_neu[c] += 1

    # 2) Assign positives by ascending label-count (easiest first)
    current_pos = np.zeros((C, K), int)
    assign = {}

    # Order easiest samples first (fewest positives)
    order = pos_idx[np.argsort(Y[pos_idx].sum(axis=1))]
    for idx in order:
        mask = Y[idx]
        pos_labels = np.where(mask == 1)[0]

        # find feasible clients (all capacities > current)
        feas = [
            c
            for c in range(C)
            if all(current_pos[c, j] < cap_pos[c, j] for j in pos_labels)
        ]

        if feas:
            # choose by distance-to-target
            best_score, c_best = None, None
            for c in feas:
                new_vec = current_pos[c] + mask
                se = ((new_vec - cap_pos[c]) ** 2).sum()
                if c_best is None or se < best_score:
                    best_score, c_best = se, c
        else:
            # fallback: pick client with max remaining label slots
            best_cnt, c_best = -1, None
            for c in range(C):
                cnt = sum(current_pos[c, j] < cap_pos[c, j] for j in pos_labels)
                if cnt > best_cnt:
                    best_cnt, c_best = cnt, c

        assign[idx] = c_best
        # decrement only where capacity remains
        for j in pos_labels:
            if current_pos[c_best, j] < cap_pos[c_best, j]:
                current_pos[c_best, j] += 1

    # 3) Assign neutrals
    current_neu = np.zeros(C, int)
    for idx in neu_idx:
        rems = cap_neu - current_neu
        c_best = int(np.argmax(rems))
        assign[idx] = c_best
        current_neu[c_best] += 1

    # 4) Build result DataFrame
    df2 = df.copy().reset_index(drop=True)
    df2["client"] = df2.index.map(assign)

    # 6) Ensure min_samples per client by reassigning neutrals
    assigned_counts = df2["client"].value_counts().to_dict()

    # identify clients with too few or surplus neutrals
    low_clients = [c for c in range(C) if assigned_counts.get(c, 0) < min_sample_number]
    # recompute per-client neutral indices
    neu_by_client = {c: [i for i in neu_idx if assign[i] == c] for c in range(C)}

    for c_low in low_clients:
        need = min_sample_number - assigned_counts.get(c_low, 0)
        # donors: clients with surplus neutrals
        donors = sorted(range(C), key=lambda c: len(neu_by_client[c]), reverse=True)
        for donor in donors:
            if donor == c_low:
                continue
            while need > 0 and neu_by_client[donor]:
                idx_swap = neu_by_client[donor].pop()
                assign[idx_swap] = c_low
                df2.at[idx_swap, "client"] = c_low
                current_neu[donor] -= 1
                current_neu[c_low] += 1
                assigned_counts[donor] -= 1
                assigned_counts[c_low] = assigned_counts.get(c_low, 0) + 1
                need -= 1
            if need <= 0:
                break

    if verbose:
        print_df_distribution(df2, K, C, pathology_names)
    return df2


def train_val_split(df, train_val_prop, random_state):
    df = df.copy()
    is_multilabel = (
        isinstance(df["target"].iloc[0], list) and len(df["target"].iloc[0]) > 1
    )

    if is_multilabel:
        df.loc[:, "strat_target"] = df["target"].apply(lambda x: tuple(x))
    else:
        df.loc[:, "strat_target"] = df["target"]

    value_counts = df["strat_target"].value_counts()
    major_keys = value_counts[value_counts >= 2].index
    major_classes_df = df[df["strat_target"].isin(major_keys)].copy()
    minor_classes_df = df[~df["strat_target"].isin(major_keys)].copy()
    n_major_classes = len(major_keys)

    # If not enough data for stratification, fall back
    if (
        len(major_classes_df) == 0
        or train_val_prop * len(major_classes_df) < n_major_classes
    ):
        # fallback to unstratified split on entire df
        train_df, valid_df = train_test_split(
            major_classes_df,
            test_size=train_val_prop,
            random_state=random_state,
        )
        train_df = pd.concat([train_df, minor_classes_df], ignore_index=True)

        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

    stratify = major_classes_df["strat_target"]
    min_freq = stratify.value_counts().min()
    max_allowed_ratio = (min_freq - 1) / min_freq
    if max_allowed_ratio < train_val_prop:
        train_val_prop = max_allowed_ratio

    train_df, valid_df = train_test_split(
        major_classes_df,
        test_size=train_val_prop,
        stratify=stratify,
        random_state=random_state,
    )
    train_df = pd.concat([train_df, minor_classes_df], ignore_index=True)

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def print_df_distribution(df, num_classes, num_clients, pathology_names=None):
    is_multilabel = isinstance(df["target"].iloc[0], list)
    if is_multilabel:
        df["class_target"] = df["target"].apply(
            lambda x: [i for i, val in enumerate(x) if val == 1] or [-1]
        )

    print(f"Total usage data: {len(df[df['client'] != -1])}")

    client_groups = df.groupby("client")
    valid_clients = set(df["client"].unique())

    for cl in range(num_clients):
        if cl not in valid_clients:
            print(f"Client {cl:>2} | No data")
            continue

        client_data = client_groups.get_group(cl)
        if is_multilabel:
            distr = (
                client_data["class_target"]
                .explode()
                .value_counts()
                .reindex(range(-1, num_classes), fill_value=0)
                .tolist()
            )
        else:
            distr = (
                client_data["target"]
                .value_counts()
                .reindex(range(num_classes), fill_value=0)
                .tolist()
            )
        x_str = " ".join(f"{num:>4}" for num in distr)
        if cl == 0 and is_multilabel:
            pathology_str = " " * 11 + "  ".join(["Other"] + pathology_names)
            print(pathology_str)
        print(f"Client {cl:>2} | {x_str} | {len(client_data):>5}")


def create_sharded_df(
    df: pd.DataFrame,
    n_clusters: int,
    amount_of_clients: int,
    dominant_ratio: float = 0.8,
    random_state: int = 42,
):
    """
    Assigns each sample in df to a client for federated learning, grouping clients into clusters.
    Each cluster has a dominant class with given dominant_ratio within the cluster,
    and clients in a cluster share roughly equal distribution of that cluster's quota.

    Ensures all data in df is used and that class counts match exactly.

    Returns:
        df with new 'client' column (integers 0..amount_of_clients-1),
        info_list: list of tuples (client_id, cluster_id, class_distribution_list)
    """
    # Copy and setup
    df = df.copy().reset_index(drop=True)
    np.random.seed(random_state)

    # Unique classes and counts
    classes = sorted(df["target"].unique())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    counts_per_class = df["target"].map(class_to_idx).value_counts().sort_index().values

    total_points = len(df)

    # Determine sizes per client
    base_size = total_points // amount_of_clients
    sizes = np.full(amount_of_clients, base_size, dtype=int)
    rem = total_points - base_size * amount_of_clients
    sizes[:rem] += 1

    # Assign clients to clusters evenly
    cpb = amount_of_clients // n_clusters
    extra = amount_of_clients - cpb * n_clusters
    cluster_of_client = []
    for cl in range(n_clusters):
        cnt = cpb + (1 if cl < extra else 0)
        cluster_of_client += [cl] * cnt
    cluster_of_client = np.array(cluster_of_client)

    # Cluster sizes
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for cl in range(n_clusters):
        cluster_sizes[cl] = sizes[cluster_of_client == cl].sum()

    # Determine cluster-level quotas per class (floats)
    cluster_quota_f = np.zeros((n_clusters, n_classes), dtype=float)
    for cl in range(n_clusters):
        # dominant class is cl mod n_classes
        dom = cl % n_classes
        ratios = np.full(n_classes, (1 - dominant_ratio) / (n_classes - 1))
        ratios[dom] = dominant_ratio
        cluster_quota_f[cl] = ratios * cluster_sizes[cl]

    # Adjust cluster quotas to match actual class counts
    cluster_quota = np.zeros_like(cluster_quota_f, dtype=int)
    for cls in range(n_classes):
        fvals = cluster_quota_f[:, cls]
        floors = np.floor(fvals).astype(int)
        frac = fvals - floors
        need = counts_per_class[cls] - floors.sum()
        # Distribute remainders
        order = np.argsort(-frac)
        add = np.zeros(n_clusters, dtype=int)
        if need > 0:
            add[order[:need]] = 1
        elif need < 0:
            # Too many assigned: subtract from smallest frac
            order_low = np.argsort(frac)
            for i in order_low[:(-need)]:
                floors[i] -= 1
        cluster_quota[:, cls] = floors + add

    # Now distribute cluster quotas to clients
    distr = np.zeros((amount_of_clients, n_classes), dtype=int)
    for cl in range(n_clusters):
        clients = np.where(cluster_of_client == cl)[0]
        szs = sizes[clients]
        csize = cluster_sizes[cl]
        for cls in range(n_classes):
            cq = cluster_quota[cl, cls]
            # ideal per-client
            ideal = cq * szs / csize
            fl = np.floor(ideal).astype(int)
            frac = ideal - fl
            need = cq - fl.sum()
            order = np.argsort(-frac)
            add = np.zeros_like(fl)
            if need > 0:
                add[order[:need]] = 1
            elif need < 0:
                order_low = np.argsort(frac)
                for i in order_low[:(-need)]:
                    fl[i] -= 1
            distr[clients, cls] = fl + add

    # Assign client labels
    df["client"] = -1
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    rem_counts = {
        i: {cls: int(distr[i, cls]) for cls in range(n_classes)}
        for i in range(amount_of_clients)
    }
    for idx, row in df.iterrows():
        cls = class_to_idx[row["target"]]
        # find a client in random order that still needs this class
        for client in np.random.permutation(amount_of_clients):
            if rem_counts[client][cls] > 0:
                df.at[idx, "client"] = client
                rem_counts[client][cls] -= 1
                break

    # Build info list
    info_list = []
    for i in range(amount_of_clients):
        info_list.append((i, int(cluster_of_client[i]), distr[i].tolist()))

    return df, info_list
