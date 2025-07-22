from ..base.base_client import BaseClient

# from utils.data_utils import get_augmentation
from omegaconf import open_dict


class PerClient(BaseClient):
    def create_pipe_commands(self):
        pipe_commands_map = super().create_pipe_commands()
        pipe_commands_map["strategy"] = self.map_client_strategy
        return pipe_commands_map

    def map_client_strategy(self, strategy):
        if strategy == "strong_filter":
            transforms = [
                {
                    "_target_": "ecglib.preprocessing.preprocess.ButterworthFilter",
                    "filter_type": "lowpass",
                    "fs": 500,
                    "n": 4,
                    "Wn": 0.3,
                },
                {
                    "_target_": "ecglib.preprocessing.preprocess.ButterworthFilter",
                    "filter_type": "highpass",
                    "fs": 500,
                    "n": 4,
                    "Wn": 0.001,
                },
                {
                    "_target_": "ecglib.preprocessing.preprocess.IIRNotchFilter",
                    "fs": 500,
                    "w0": 0.1,
                },
                {
                    "_target_": "ecglib.preprocessing.preprocess.ButterworthFilter",
                    "filter_type": "lowpass",
                    "fs": 500,
                    "n": 4,
                    "Wn": 0.05,
                },
            ]
            # with open_dict(self.cfg.ecg_record_params.augmentation):
            #     self.cfg.ecg_record_params.augmentation.prob = 1.0
            #     self.cfg.ecg_record_params.augmentation.transforms = transforms
            # augmentation = get_augmentation(self.cfg)
            # self.train_loader.dataset.augmentation = augmentation
            # self.valid_loader.dataset.augmentation = augmentation

        else:
            pass  # origin strategy remain the same client logic
