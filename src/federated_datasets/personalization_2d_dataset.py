import numpy as np
import pandas as pd
import torch

from .federated_dataset import FederatedDataset


def _make_features(
    targets, contexts, invariant_strength, local_strength, noise_std, rng
):
    signs = 2 * targets - 1
    return np.column_stack(
        [
            invariant_strength * signs + rng.normal(scale=noise_std, size=len(signs)),
            local_strength * contexts * signs
            + rng.normal(scale=noise_std, size=len(signs)),
        ]
    )


class Personalization2DDataset(FederatedDataset):
    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        samples_per_client,
        num_test_samples,
        invariant_strength,
        local_strength,
        noise_std,
        **kwargs,
    ):
        self.samples_per_client = samples_per_client
        self.num_test_samples = num_test_samples
        self.invariant_strength = invariant_strength
        self.local_strength = local_strength
        self.noise_std = noise_std
        super().__init__(cfg, mode, data_sources, base_path)

    def df_exist(self):
        return True

    def load_map_files(self):
        rng = np.random.RandomState(self.cfg.random_state + (self.mode == "test"))
        if self.mode == "train":
            return self._make_client_data(rng)
        return self._make_test_data(rng)

    def _make_client_data(self, rng):
        frames = []
        for client in range(self.cfg.federated_params.amount_of_clients):
            targets = np.arange(self.samples_per_client) % 2
            rng.shuffle(targets)
            context = self._client_contexts()[client]
            contexts = np.full(self.samples_per_client, context)
            features = _make_features(
                targets,
                contexts,
                self.invariant_strength,
                self.local_strength,
                self.noise_std,
                rng,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "x1": features[:, 0],
                        "x2": features[:, 1],
                        "target": targets,
                        "client": client,
                        "context": contexts,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def _make_test_data(self, rng):
        targets = np.arange(self.num_test_samples) % 2
        client_contexts = self._client_contexts()
        contexts = client_contexts[
            np.arange(self.num_test_samples) // 2 % len(client_contexts)
        ]
        order = rng.permutation(self.num_test_samples)
        targets = targets[order]
        contexts = contexts[order]
        features = _make_features(
            targets,
            contexts,
            self.invariant_strength,
            self.local_strength,
            self.noise_std,
            rng,
        )
        return pd.DataFrame(
            {
                "x1": features[:, 0],
                "x2": features[:, 1],
                "target": targets,
                "context": contexts,
            }
        )

    def _client_contexts(self):
        count = self.cfg.federated_params.amount_of_clients
        magnitudes = np.linspace(0.8, 1.2, (count + 1) // 2)
        return np.array(
            [
                magnitudes[client // 2] * (1 if client % 2 == 0 else -1)
                for client in range(count)
            ]
        )

    def split_to_clients(self):
        print("Used predefined alternating client contexts")

    def __getitem__(self, index):
        row = self.data.iloc[index]
        features = torch.tensor([row.x1, row.x2], dtype=torch.float32)
        return index, ([features], int(row.target))
