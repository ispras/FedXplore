import os
import ast
import hydra
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from ecglib.data.load_datasets import load_ptb_xl
from ecglib.data.datasets import EcgDataset
from ecglib.preprocessing.composition import Compose

from .federated_dataset import FederatedDataset
from utils.data_utils import create_dirichlet_multilabel_df
from utils.dataset_utils import get_target_dir, set_data_configs, save_map_files


class PTBXLDataset(FederatedDataset):
    def __init__(self, cfg, mode, data_sources, dataset_cfg):
        self.dataset_cfg = dataset_cfg
        # if the current dataset is the same as the train dataset,
        # then we need to copy the data configs.
        # This is necessary to avoid setting the same parameters in all datasets.
        if (
            cfg.train_dataset["_target_"]
            == cfg.__getattr__(f"{mode}_dataset")["_target_"]
        ):
            self.dataset_cfg = cfg.train_dataset.dataset_cfg

        super().__init__(cfg, mode, data_sources)
        if self.mode != "train":
            self.init_ecg_dataset()

    def downloading(self):
        print("We will download PTB-XL dataset!")
        print(
            "You can restart the experiment and set `+train_dataset.download_path` "
            "to specify the path to save the dataset."
        )
        # You can configure the folder for saving the dataset by adding:
        # +train_dataset.download_path // +test_dataset.download_path to run command
        # By default, the root folder of the project will be used
        target_dir = get_target_dir(self.cfg, default_dir="ptbxl")

        # 1. Download dataset
        ptbxl_df = load_ptb_xl(
            download=True,
            path_to_zip=f"{target_dir}_zip",
            path_to_unzip=target_dir,
            delete_zip=True,
            frequency=500,
        )
        # 2. Update instantiated Dataset class
        self.target_dir = os.path.join(
            target_dir,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        )
        super().downloading()
        # 3. Save map-files in target directory
        train_df = ptbxl_df[ptbxl_df["strat_fold"] != 10]
        test_df = ptbxl_df[ptbxl_df["strat_fold"] == 10]
        save_map_files(train_df, test_df, self.target_dir)
        # 4. Update paths in yaml config
        set_data_configs(self.target_dir, config_names=["ptbxl.yaml"])

    def preprocessing(self):
        # 1. Update df according to frequency
        self.data["frequency"] = self.dataset_cfg.frequency
        if self.dataset_cfg.frequency != 500:
            assert (
                self.dataset_cfg.frequency == 100
            ), f"PTB-XL signals are only supported with 100 or 500 sample frequency, recieved: {self.dataset_cfg.frequency}"
            suffix = "lr"  # low rate
            self.data["fpath"] = [
                self.data.iloc[i][f"fpath"].split(".")[0][:-2]
                + suffix
                + "."
                + self.data.iloc[i][f"fpath"].split(".")[1]
                for i in range(len(self.data))
            ]

        # 2. Remove NaN labels
        self.data = self.data[self.data[self.dataset_cfg.task_type].notna()]
        # Remove artefact file
        self.data = self.data[self.data["filename_lr"] != "records100/12000/12722_lr"]

        # 3. Use only trusted labels
        if self.dataset_cfg.validated_by_human:
            self.data = self.data[self.data["validated_by_human"]]

        # 4. Form target labels
        self.data.reset_index(drop=True, inplace=False)
        one_hot = make_onehot(
            self.data, self.dataset_cfg.task_type, self.dataset_cfg.pathology_names
        )
        if self.dataset_cfg.merge_map:
            one_hot = merge_columns(df=one_hot, merge_map=self.dataset_cfg.merge_map)
        self.data["target"] = one_hot.values.tolist()

    def split_to_clients(self):
        if self.distribution.name == "dirichlet_multilabel":
            self.data = create_dirichlet_multilabel_df(
                self.data,
                self.cfg.federated_params.amount_of_clients,
                self.distribution.alpha,
                self.distribution.verbose,
                self.distribution.min_sample_number,
                self.cfg.random_state,
                self.dataset_cfg.pathology_names,
            )
        else:
            # HARDCODE
            # If not Dirichlet, we gonna do uniform distribution
            # But with Dirichlet distribution with alpha=1000

            self.data = create_dirichlet_multilabel_df(
                self.data,
                self.cfg.federated_params.amount_of_clients,
                alpha=1000,
                verbose=True,
                min_sample_number=20,
                seed=self.cfg.random_state,
                pathology_names=self.dataset_cfg.pathology_names,
            )

    def get_augmentation(self):
        augmentation_transform = self.dataset_cfg.augmentation.transforms
        if augmentation_transform:
            aug_list = [
                hydra.utils.instantiate(augm) for augm in augmentation_transform
            ]
            augmentation = Compose(
                transforms=aug_list, p=self.dataset_cfg.augmentation.prob
            )
        else:
            augmentation = None
        return augmentation

    def init_ecg_dataset(self):
        self.ecg_dataset = EcgDataset(
            ecg_data=self.data,
            target=self.data.target.values,
            frequency=self.dataset_cfg.frequency,
            leads=list(self.dataset_cfg.leads),
            norm_type=self.dataset_cfg.norm_type,
            classes=self.num_classes,
            augmentation=self.get_augmentation(),
        )

    def to_client_side(self, rank):
        self.rank = rank
        self.data = self.orig_data[self.orig_data["client"] == rank]
        self.init_ecg_dataset()
        return self

    def __getitem__(self, index):
        return self.ecg_dataset.__getitem__(index)


def make_onehot(ecg_df, task_type, pathology_names=None):
    """
    Create one_hot vectors for classification

    :param ecg_df: input dataframe
    :param task_type: type of predicted classes (registers, syndromes, etc.)
    :param pathology_names: list of predicted classes

    :return: pandas dataframe
    """
    mlb = MultiLabelBinarizer()
    one_hot = pd.DataFrame(
        mlb.fit_transform(ecg_df[task_type].apply(ast.literal_eval)),
        columns=mlb.classes_,
    )
    if pathology_names:
        drop_cols = set(one_hot.columns) - set(pathology_names)
        one_hot.drop(columns=drop_cols, inplace=True)
    return one_hot


def merge_columns(df, merge_map):
    """
    Logical OR for given one-hot columns

    :param df: input dataframe
    :param merge_map: dictionary: key - name after merge, value - list of columns to be merged

    :return: pandas DataFrame
    """
    for k, v in merge_map.items():

        existing_columns = set(v).intersection(set(df.columns))
        assert (
            len(existing_columns) != 0
        ), f"None of the specified pathologies {v} exist in the dataset."

        if existing_columns != set(v):
            print(
                f"Pathologies do not exist in the dataset: {set(v) - set(df.columns)}. Using only existing pathologies: {existing_columns}."
            )

        tmp = df[list(existing_columns)].apply(any, axis=1).astype(int)
        df.drop(columns=existing_columns, inplace=True)
        df[k] = tmp
    return df
