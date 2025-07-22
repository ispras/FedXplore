import os
import warnings
import numpy as np
import pandas as pd
from omegaconf import open_dict
from torch.utils.data import Dataset
from utils.data_utils import create_dirichlet_df, train_val_split


class FederatedDataset(Dataset):
    def __init__(self, cfg, mode, data_sources):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.distribution = cfg.distribution
        self.data_sources = data_sources
        self.name = self.__class__.__name__
        self.init_df()

    def init_df(self):
        if not self.df_exist():
            self.downloading()

        self.data = self.load_map_files()
        self.preprocessing()

        if "trust_dataset" in self.cfg:
            self.parse_trust()

        self.define_num_classes()

        if self.mode == "train":
            print(f"Used distribution is: {self.distribution.name}")
            print(f"Full params: {self.distribution}\n")
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

    def load_map_files(self):
        df = pd.DataFrame()
        for directories in self.data_sources[f"{self.mode}_map_file"]:
            df = pd.concat([df, pd.read_csv(directories, low_memory=False)])

        return df

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
            if self.mode == "train":  # print once
                print(f"We will trust dataset with {num_trust_samples} by default.")
                print(
                    f"You can configure this number by setting `+trust_dataset.num_trust_samples=number`\n"
                )

        train_val_prop = num_trust_samples / len(df)

        train_df, trust_df = train_val_split(
            df, train_val_prop, random_state=self.cfg.random_state
        )

        # Setup save directory
        save_dir = os.path.dirname(self.data_sources["train_map_file"][0])

        # Save new train map file
        save_path = os.path.join(save_dir, "train_without_trust_map_file.csv")
        train_df.to_csv(save_path)
        with open_dict(self.cfg):
            self.cfg.train_dataset.data_sources.train_map_file = [save_path]

        # Update train paths in this instantiated class
        self.data_sources.train_map_file = [save_path]

        # Save new trust map file
        save_path = os.path.join(save_dir, "trust_map_file.csv")
        trust_df.to_csv(save_path)
        with open_dict(self.cfg):
            self.cfg.trust_dataset.data_sources.trust_map_file = [save_path]

        self.data = train_df

    def separate_trust_df(self):
        warnings.warn(
            f"As trust dataset used train part of data "
            f"To change this behavior, rewrite the function "
            f"get_separate_trust_df in {self.name} dataset"
        )

        # Basically we will use the train part of our dataset as trust
        # But you can configure this separately for each dataset
        self.mode = "train"

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
        if self.distribution.name == "dirichlet":
            self.data = create_dirichlet_df(
                self.data,
                self.cfg.federated_params.amount_of_clients,
                self.distribution.alpha,
                self.distribution.verbose,
                self.cfg.random_state,
            )
        else:
            # HARDCODE
            # If not Dirichlet, we gonna do uniform distribution
            # But with Dirichlet distribution with alpha=1000

            self.data = create_dirichlet_df(
                self.data,
                self.cfg.federated_params.amount_of_clients,
                alpha=1000,
                verbose=True,
                random_state=self.cfg.random_state,
            )

    def to_client_side(self, rank):
        self.rank = rank
        self.data = self.orig_data[self.orig_data["client"] == rank]

        return self

    def get_cfg(self):
        return self.cfg

    def __getitem__(self, index):
        raise NotImplementedError(
            f"You need to implement __getitem__ function " f"in {self.name} dataset!"
        )

    def __len__(self):
        return len(self.data)
