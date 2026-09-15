import numpy as np
import pandas as pd
import torch

from .federated_dataset import FederatedDataset


def make_gaussian_2d_dataframe(num_samples, class_distance, noise_std, random_state):
    rng = np.random.RandomState(random_state)
    targets = np.arange(num_samples) % 2
    rng.shuffle(targets)

    class_signs = 2 * targets - 1
    centers = np.column_stack([class_signs, class_signs]) * class_distance / 2
    features = centers + rng.normal(scale=noise_std, size=(num_samples, 2))

    return pd.DataFrame(
        {
            "x1": features[:, 0],
            "x2": features[:, 1],
            "target": targets,
        }
    )


class Synthetic2DDataset(FederatedDataset):
    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        num_train_samples,
        num_test_samples,
        class_distance,
        noise_std,
        **kwargs,
    ):
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.class_distance = class_distance
        self.noise_std = noise_std
        super().__init__(cfg, mode, data_sources, base_path)

    def df_exist(self):
        return True

    def load_map_files(self):
        num_samples = (
            self.num_train_samples if self.mode == "train" else self.num_test_samples
        )
        seed_offset = 0 if self.mode == "train" else 1
        return make_gaussian_2d_dataframe(
            num_samples,
            self.class_distance,
            self.noise_std,
            self.cfg.random_state + seed_offset,
        )

    def __getitem__(self, index):
        row = self.data.iloc[index]
        features = torch.tensor([row.x1, row.x2], dtype=torch.float32)
        return index, ([features], int(row.target))
