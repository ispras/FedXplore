import numpy as np
from utils.data_utils import print_df_distribution


class DirichletMultilabelDistribution:
    def __init__(self, alpha, verbose, min_sample_number):
        self.alpha = alpha
        self.verbose = verbose
        self.min_sample_number = min_sample_number

    def split_to_clients(self, df, amount_of_clients, random_state, pathology_names):
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
        rng = np.random.RandomState(random_state)
        dir_samples = rng.dirichlet([self.alpha] * C, size=K)
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
        rng2 = np.random.RandomState(random_state + 1)
        p_neu = rng2.dirichlet([self.alpha] * C)
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
        low_clients = [
            c for c in range(C) if assigned_counts.get(c, 0) < self.min_sample_number
        ]
        # recompute per-client neutral indices
        neu_by_client = {c: [i for i in neu_idx if assign[i] == c] for c in range(C)}

        for c_low in low_clients:
            need = self.min_sample_number - assigned_counts.get(c_low, 0)
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

        if self.verbose:
            print_df_distribution(df2, K, C, pathology_names)
        return df2
