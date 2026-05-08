import hydra
import pandas as pd
import ast
import os
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from ecglib.data.datasets import EcgDataset
from ecglib.preprocessing.composition import Compose

from .federated_dataset import FederatedDataset

from utils.dataset_utils import ZarrEcgDataset
from .ptbxl_dataset import make_onehot, merge_columns


class TISDataset(FederatedDataset):
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

        assert (
            not self.dataset_cfg.use_metadata
        ), f"At the current moment, we don't take metadata into account. You set self.dataset_cfg.use_metadata={self.dataset_cfg.use_metadata}"

        distribution_name = cfg.distribution._target_.split(".")[-1]
        supported_distributions = [
            "DirichletMultilabelDistribution",
            "UniformDistribution",
            "HospitalDistribution",
        ]
        assert (
            distribution_name in supported_distributions
        ), f"For ECG multilabel scenario only {supported_distributions} are allowed. You provided: {distribution_name}"

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

        super().__init__(cfg, mode, data_sources, base_path)
        self.init_ecg_dataset()

    def downloading(self):
        raise Exception("Proprietary TIS could not be downloaded.")

    def preprocessing(self):
        # 0. we preprocessed data on train stage, so, for trust it is already preprocessed
        if self.loading_mode == "trust":
            self.data["target"] = self.data["target"].apply(
                lambda x: ast.literal_eval(x)
            )
            return
        # 1. Remove rows with filenames listed in filter_dataframe_files
        self.filter_by_another_dataframes()
        # 2. Remove NaN labels
        self.data = self.data[self.data["scp_codes"].notna()]
        # 3. Filter by frequency
        self.filter_by_frequency()
        # 4. Filter by frequency
        self.filter_by_length_range()
        # 5. Patient metadata preprocessing
        self.data["patient_metadata"] = self.data["patient_metadata"].apply(
            lambda x: ast.literal_eval(x)
        )
        # 6. Filter by patient age range
        self.filter_by_age_range()
        # 7. if metadata is not required than we delete it because ecglib can handle its absence
        if not self.dataset_cfg.use_metadata:
            self.filter_metadata()
        # 8. Form target labels
        self.data.reset_index(drop=True, inplace=False)
        one_hot = make_onehot(self.data, "scp_codes", self.dataset_cfg.pathology_names)
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
                hydra.utils.instantiate(augm["config"], _convert_="all")
                for augm in augmentation_transform
            ]
            augmentation = Compose(
                transforms=aug_list, p=self.dataset_cfg.augmentation.augm_prob
            )
        else:
            augmentation = None
        return augmentation

    def dataset_split(self, train_val_prop):
        valid_dataset = super().dataset_split(train_val_prop)
        # init ecg dataset for valid part
        valid_dataset.init_ecg_dataset()
        # init ecg dataset for train part
        self.init_ecg_dataset()
        return valid_dataset

    def init_ecg_dataset(self):
        allowed_data_types = {"npz", "zarr"}
        assert (
            self.dataset_cfg.data_type in allowed_data_types
        ), f"Available set of data types: {allowed_data_types}, you provide: {self.dataset_cfg.data_type}"
        if self.dataset_cfg.data_type == "zarr":
            assert (
                OmegaConf.select(self.data_sources, "path_to_zarr") is not None
            ), "if `data_type`=='zarr', you need to provide a `path_to_zarr` inside dataset.data_sources config"

        if self.dataset_cfg.data_type != "zarr":
            self.ecg_dataset = EcgDataset(
                ecg_data=self.data,
                target=self.data.target.values,
                frequency=self.dataset_cfg.resampled_frequency,
                leads=list(self.dataset_cfg.leads),
                data_type=self.dataset_cfg.data_type,
                ecg_length=self.dataset_cfg.resampled_ecg_length,
                cut_range=self.dataset_cfg.ecg_cut_range,
                norm_type=self.dataset_cfg.norm_type,
                classes=self.num_classes,
                augmentation=self.get_augmentation(),
            )
        else:
            self.ecg_dataset = ZarrEcgDataset(
                ecg_data=self.data,
                target=self.data.target.values,
                frequency=self.dataset_cfg.resampled_frequency,
                leads=list(self.dataset_cfg.leads),
                data_type=self.dataset_cfg.data_type,
                ecg_length=self.dataset_cfg.resampled_ecg_length,
                cut_range=self.dataset_cfg.ecg_cut_range,
                norm_type=self.dataset_cfg.norm_type,
                classes=self.num_classes,
                augmentation=self.get_augmentation(),
                path_to_zarr=self.data_sources.path_to_zarr[0],
            )

    def __getitem__(self, index):
        return self.ecg_dataset.__getitem__(index)

    def filter_by_another_dataframes(self):
        """
        filter the dataframe by name of files from others

        :return: filtered Dataframe
        """
        for file_path in self.data_sources["filter_map_files"]:
            filter_df = pd.read_csv(file_path, low_memory=False)
            filter_df_list = filter_df.file_name.to_list()
            self.data = self.data[~self.data.file_name.isin(filter_df_list)]

    def filter_by_age_range(self):
        self.data = self.data[
            (
                (self.data.age >= self.dataset_cfg.age_range[0])
                & (self.data.age <= self.dataset_cfg.age_range[1])
            )
        ]

    def filter_by_frequency(self):
        self.data = self.data[
            self.data["frequency"].isin(self.dataset_cfg.observed_frequencies)
        ]

    def filter_by_length_range(self):
        self.data = self.data[
            (
                (self.data.ecg_duration >= self.dataset_cfg.observed_ecg_length[0])
                & (self.data.ecg_duration <= self.dataset_cfg.observed_ecg_length[1])
            )
        ]

    def filter_metadata(self):
        self.data = self.data.drop(columns=["ecg_metadata"])
        self.data = self.data.drop(columns=["patient_metadata"])
