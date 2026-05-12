import os
import warnings
import copy
import numpy as np
import pandas as pd
from omegaconf import open_dict
from torch.utils.data import Dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from hydra.core.hydra_config import HydraConfig

from utils.dataset_utils import update_data_sources


class FederatedDataset(Dataset):
    def __init__(self, cfg, mode, data_sources, base_path):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.distribution = instantiate(cfg.distribution)
        self.data_sources = data_sources

        assert (
            base_path == cfg.train_dataset.base_path
        ), f"We need to duplicate `base_path` changes to all considering datasets. You {self.mode} dataset contains base_path={base_path}"

        if self.df_exist():
            self.data_sources = update_data_sources(base_path, data_sources)
        self.name = self.__class__.__name__
        self.init_df()

    def init_df(self):
        if not self.df_exist():
            self.downloading()

        self.loading_mode = self.get_loading_mode()
        self.data = self.load_map_files()
        self.preprocessing()

        if "trust_dataset" in self.cfg:
            self.parse_trust()

        self.define_num_classes()

        if self.mode == "train":
            self.split_to_clients()

        # Save original df, we will use it after reinit on client side
        self.orig_data = self.data

    def df_exist(self):
        # Custom mode of the dataset are not intended for downloading by default
        # For example, the trust dataset is usually part of the train in our implementation
        # Therefore this function returns `True` will see some unknown mod name
        if f"{self.mode}_map_file" not in self.data_sources:
            return True

        # Сheck if the necessary train/test paths exist
        return self.data_sources[f"{self.mode}_map_file"] is not None

    def downloading(self):
        # Abstract pipeline for automatic downloading of new datasets.
        # It assumes the structure with `train_map_file.csv` and `test_map_file.csv`
        assert getattr(
            self, "target_dir", False
        ), "To update the dataset during download, the `self.target_dir` attribute is required."

        train_map_path = os.path.join(self.target_dir, "train_map_file.csv")
        test_map_path = os.path.join(self.target_dir, "test_map_file.csv")
        self.data_sources["train_map_file"] = [train_map_path]
        self.data_sources["test_map_file"] = [test_map_path]
        with open_dict(self.cfg):
            self.cfg.train_dataset.data_sources.train_map_file = [train_map_path]
            self.cfg.test_dataset.data_sources.test_map_file = [test_map_path]

    def get_loading_mode(self):
        loading_mode = self.mode

        if self.mode == "trust":
            if "trust_map_file" in self.cfg.trust_dataset.data_sources:
                loading_mode = "trust"
            else:
                if hasattr(self.cfg.trust_dataset, "trust_base_data_part"):
                    self.trust_base_data_part = (
                        self.cfg.trust_dataset.trust_base_data_part
                    )
                else:
                    print(f"We will use train as trust dataset by default.")
                    print(
                        f"You can configure this by setting `+trust_dataset.trust_base_data_part=<train/test>`\n"
                    )
                    self.trust_base_data_part = "train"

                loading_mode = self.trust_base_data_part
        return loading_mode

    def load_map_files(self):
        df = pd.DataFrame()
        for directories in self.data_sources[f"{self.loading_mode}_map_file"]:
            df = pd.concat([df, pd.read_csv(directories, low_memory=False)])

        return df

    def parse_trust(self):
        # Parse trust dataset case (additionaly data on server side)
        if self.cfg.train_dataset["_target_"] == self.cfg.trust_dataset["_target_"]:
            if self.mode == "train":
                # If trust dataset is same as train dataset
                # So we need to get part of train data to trust
                return self.repeating_trust_df()
        else:
            if self.mode == "trust":
                # Trust dataset is another dataset from train
                # So we can use any part of data for trust
                # Which part to use will configure in get_separate_trust_df
                return self.separate_trust_df()

    def repeating_trust_df(self):
        # Load all train data
        df = self.data

        # By default we use 500 examples for trust
        # But you can configure that by
        # +trust_dataset.num_trust_samples=number
        # in your run command
        num_trust_samples = 500
        if "num_trust_samples" in self.cfg.trust_dataset:
            num_trust_samples = self.cfg.trust_dataset.num_trust_samples
        else:
            print(f"We will trust dataset with {num_trust_samples} by default.")
            print(
                f"You can configure this number by setting `+trust_dataset.num_trust_samples=number`\n"
            )

        val_prop = num_trust_samples / len(df)

        train_df, trust_df = self.train_val_split(
            df, val_prop, random_state=self.cfg.random_state
        )

        # Setup save directory
        save_dir = HydraConfig.get().runtime.output_dir

        # Save new trust map file
        save_path = os.path.join(save_dir, "trust_map_file.csv")
        trust_df.to_csv(save_path)
        print(f"New trust map-file saved in: {save_path}\n")

        with open_dict(self.cfg):
            self.cfg.trust_dataset.data_sources.trust_map_file = [save_path]

        self.data = train_df

    def separate_trust_df(self):
        df = self.data
        num_trust_samples = 500
        if "num_trust_samples" in self.cfg.trust_dataset:
            num_trust_samples = self.cfg.trust_dataset.num_trust_samples
        else:
            print(f"We will trust dataset with {num_trust_samples} by default.")
            print(
                f"You can configure this number by setting `+trust_dataset.num_trust_samples=number`\n"
            )

        train_val_prop = num_trust_samples / len(df)

        _, trust_df = self.train_val_split(
            df, train_val_prop, random_state=self.cfg.random_state
        )

        # Save new trust map file
        save_dir = HydraConfig.get().runtime.output_dir
        save_path = os.path.join(save_dir, "trust_map_file.csv")
        print(f"New trust map-file saved in: {save_path}\n")
        trust_df.to_csv(save_path)
        with open_dict(self.cfg):
            self.cfg.trust_dataset.data_sources.trust_map_file = [save_path]

        self.data = trust_df

    def preprocessing(self):
        pass

    def define_num_classes(self):
        if isinstance(self.data.iloc[0]["target"], list):
            # Multilabel case
            self.num_classes = len(self.data.iloc[0]["target"])
        else:
            self.num_classes = pd.Series(
                np.concatenate(
                    self.data["target"]
                    .apply(lambda x: x if isinstance(x, list) else [x])
                    .values
                )
            ).nunique()

        with open_dict(self.cfg):
            self.cfg.training_params.num_classes = self.num_classes
        if self.cfg.model.num_classes != self.num_classes:
            with open_dict(self.cfg):
                self.cfg.model.num_classes = self.num_classes

    def split_to_clients(self):
        print(f"Used distribution is: {self.distribution.__class__.__name__}")
        self.data = self.distribution.split_to_clients(
            self.data,
            self.cfg.federated_params.amount_of_clients,
            self.cfg.random_state,
        )

    def to_client_side(self, rank):
        self.rank = rank
        self.data = self.orig_data[self.orig_data["client"] == rank]

        return self

    def dataset_split(self, train_val_prop):
        # This method specifies valid dataset according to valid_data and sets splitted train data
        train_data, valid_data = self.train_val_split(
            self.data, train_val_prop, self.cfg.random_state
        )
        valid_dataset = copy.deepcopy(self)
        valid_dataset.data = valid_data
        valid_dataset.mode = "valid"
        self.data = train_data
        return valid_dataset

    @staticmethod
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

    def get_cfg(self):
        return self.cfg

    def __getitem__(self, index):
        raise NotImplementedError(
            f"You need to implement __getitem__ function " f"in {self.name} dataset!"
        )

    def __len__(self):
        return len(self.data)
