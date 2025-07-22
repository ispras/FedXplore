import os
from PIL import Image
from omegaconf import open_dict
import torchvision.transforms as transforms
import pandas as pd
from datasets import load_dataset
from .federated_dataset import FederatedDataset
from utils.dataset_utils import get_target_dir, save_map_files, set_data_configs


class TinyImageNet(FederatedDataset):
    def __init__(self, cfg, mode, data_sources, **kwargs):
        # Setted transform for TinyImageNet dataset
        self.transform = self.set_up_transform(mode)
        super().__init__(cfg, mode, data_sources, **kwargs)

    def set_up_transform(self, mode):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform

    def downloading(self):
        print("We will download TinyImageNet dataset!")
        print(
            "You can restart the experiment and set `+train_dataset.download_path` "
            "to specify the path to save the dataset."
        )

        # You can configure the folder for saving the dataset by adding:
        # +train_dataset.download_path // +test_dataset.download_path to run command
        # By default, the root folder of the project will be used
        target_dir = get_target_dir(self.cfg, default_dir="tiny_imagenet")

        # 1. Download dataset
        train_df, test_df = tiny_imagenet_download()
        # 2. Convert to images, create a map-file
        self.target_dir = os.path.join(target_dir, "images")
        os.makedirs(self.target_dir, exist_ok=True)
        train_df = convert_df(train_df, "train", self.target_dir)
        test_df = convert_df(test_df, "test", self.target_dir)
        # 3. Update instantiated Dataset class
        super().downloading()
        # 4. Save map-files in target directory
        save_map_files(train_df, test_df, self.target_dir)
        # 5. Update paths in yaml config
        set_data_configs(self.target_dir, config_names=["cifar10.yaml"])

    def __getitem__(self, index):
        fpath = self.data.at[index, "fpath"]
        img = Image.open(fpath)
        img_tensor = self.transform(img)
        label = self.data.at[index, "target"]
        return index, ([img_tensor], label)


def tiny_imagenet_download():
    print("Downloading TinyImageNet...")
    train_dataset = load_dataset("Maysee/tiny-imagenet", split="train")
    test_dataset = load_dataset("Maysee/tiny-imagenet", split="valid")

    print("Converting TinyImageNet...", flush=True)
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    train_df = train_df.rename(columns={"label": "target"})
    test_df = test_df.rename(columns={"label": "target"})

    return train_df, test_df


def convert_df(df, mode, save_path):
    data = []
    for idx, row in df.iterrows():
        image = row["image"]
        target = row["target"]

        transformed_img = image.convert("RGB")

        filename = f"{mode}_{idx}.png"
        file_path = os.path.join(save_path, filename)
        transformed_img.save(file_path)

        data.append({"fpath": file_path, "target": target, "client": -1})

    df = pd.DataFrame(data)
    return df
