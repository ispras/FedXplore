import os
import ast
import hydra
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from ecglib.data.load_datasets import load_ptb_xl
from ecglib.data.datasets import EcgDataset
from ecglib.preprocessing.composition import Compose

from .federated_dataset import FederatedDataset
from utils.dataset_utils import get_target_dir, set_data_configs, save_map_files


class PTBXLDataset(FederatedDataset):
    def __init__(
        self,
        cfg,
        mode,
        data_sources,
        base_path,
        dataset_cfg,
        **kwargs,
    ):
        self.dataset_cfg = dataset_cfg

        distribution_name = cfg.distribution._target_.split(".")[-1]
        supported_distributions = [
            "DirichletMultilabelDistribution",
            "UniformDistribution",
        ]
        assert (
            distribution_name in supported_distributions
        ), f"For ECG multilabel scenario only {supported_distributions} are allowed. You provided: {distribution_name}"

        assert (
            not self.dataset_cfg.use_metadata
        ), f"At the current moment, we don't take metadata into account. You set self.dataset_cfg.use_metadata={self.dataset_cfg.use_metadata}"

        assert (
            cfg.loss.loss_name == "bce"
        ), f"ECG task supports only with `BCE` loss. You set {cfg.loss.loss_name}"

        # if the current dataset is the same as the train dataset,
        # then we need to copy the data configs.
        # This is necessary to avoid setting the same parameters in all datasets.
        if (
            cfg.train_dataset["_target_"]
            == cfg.__getattr__(f"{mode}_dataset")["_target_"]
        ):
            self.dataset_cfg = cfg.train_dataset.dataset_cfg

        # PRIVATE part of code due to server shared paths
        self.shared_path = (
            "ecg_data/physionet_data/ptbxl_data_1_0_3"
            in data_sources[f"train_map_file"][0]
        )
        if self.shared_path and self.dataset_cfg.frequency == 100:
            assert (
                "ptbxl_data_sf100" in data_sources[f"{mode}_map_file"][0]
            ), f"With server shared paths sample frequency 100 placed in `ecg_data/physionet_data/ptbxl_data_sf100/`, you provided: {data_sources[f'{mode}_map_file'][0]}"

        super().__init__(cfg, mode, data_sources, base_path)
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

        # 0. Create dirs
        zip_dir = f"{target_dir}_zip"
        os.makedirs(zip_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

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
        if self.loading_mode == "trust":
            self.data["target"] = self.data["target"].apply(ast.literal_eval)
            return
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
        suffix = "lr" if self.dataset_cfg.frequency == 100 else "hr"
        # PRIVATE part of code
        if self.shared_path:
            self.data = self.data[self.data["file_name"] != f"12000_12722_{suffix}.npz"]
        else:
            self.data = self.data[
                self.data[f"filename_{suffix}"] != f"records100/12000/12722_{suffix}"
            ]

        # 3. Use only trusted labels
        if self.dataset_cfg.validated_by_human:
            self.data = self.data[self.data["validated_by_human"]]

        # 4. if metadata is not required than we delete it because ecglib can handle its absence
        if not self.dataset_cfg.use_metadata:
            self.data = self.data.drop(columns=["ecg_metadata"])
            self.data = self.data.drop(columns=["patient_metadata"])

        # 5. Form target labels
        self.data.reset_index(drop=True, inplace=False)
        one_hot = make_onehot(
            self.data, self.dataset_cfg.task_type, self.dataset_cfg.pathology_names
        )
        if self.dataset_cfg.merge_map:
            one_hot = merge_columns(df=one_hot, merge_map=self.dataset_cfg.merge_map)
        self.data["target"] = one_hot.values.tolist()

    def split_to_clients(self):
        print(f"Used distribution is: {self.distribution.__class__.__name__}")
        self.data = self.distribution.split_to_clients(
            self.data,
            self.cfg.federated_params.amount_of_clients,
            self.cfg.random_state,
            self.dataset_cfg.pathology_names,
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
            data_type="npz",
            norm_type=self.dataset_cfg.norm_type,
            classes=self.num_classes,
            augmentation=self.get_augmentation(),
        )

    def dataset_split(self, train_val_prop):
        valid_dataset = super().dataset_split(train_val_prop)
        # init ecg dataset for valid part
        valid_dataset.init_ecg_dataset()
        # init ecg dataset for train part
        self.init_ecg_dataset()
        return valid_dataset

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
