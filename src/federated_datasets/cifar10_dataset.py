import os
import tarfile
import requests
import pickle
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from .federated_dataset import FederatedDataset
from utils.dataset_utils import set_data_configs, get_target_dir, save_map_files


class Cifar10Dataset(FederatedDataset):
    def __init__(self, cfg, mode, data_sources, **kwargs):
        # Setted transform for CIFAR-10 dataset
        self.transform = self.set_up_transform(mode)
        super().__init__(cfg, mode, data_sources)

    def set_up_transform(self, mode):
        image_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if mode == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return transform

    def downloading(self):
        print("We will download CIFAR-10 dataset!")
        print(
            "You can restart the experiment and set `+train_dataset.download_path` "
            "to specify the path to save the dataset."
        )

        # You can configure the folder for saving the dataset by adding:
        # +train_dataset.download_path // +test_dataset.download_path to run command
        # By default, the root folder of the project will be used
        target_dir = get_target_dir(self.cfg, default_dir="cifar10")

        # 1. Download dataset
        download_cifar10(target_dir)
        # 2. Convert to images, create a map-file
        train_df, test_df = process_cifar10(target_dir)
        # 3. Update instantiated Dataset class
        self.target_dir = os.path.join(target_dir, "images")
        super().downloading()
        # 4. Save map-files in target directory
        save_map_files(train_df, test_df, self.target_dir)
        # 5. Update paths in yaml config
        set_data_configs(self.target_dir, config_names=["cifar10.yaml"])

    def __getitem__(self, index):
        image = Image.open(self.data["fpath"][index])
        image = self.transform(image)
        label = self.data["target"][index]
        return index, ([image], label)


def download_cifar10(target_dir="cifar10"):
    os.makedirs(target_dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(target_dir, "cifar-10-python.tar.gz")

    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            f.write(response.content)

    # Extract files
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    os.remove(tar_path)


def process_cifar10(base_dir="cifar10"):
    img_dir = os.path.join(base_dir, "images", "data")
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.isabs(img_dir):
        curent_run_path = os.getcwd()
        img_dir = os.path.join(curent_run_path, img_dir)

    with open(os.path.join(base_dir, "cifar-10-batches-py", "batches.meta"), "rb") as f:
        meta = pickle.load(f)
    label_names = meta["label_names"]

    all_data = []
    print("Converting CIFAR-10...", flush=True)
    for split in ["train", "test"]:
        files = (
            ["data_batch_%d" % i for i in range(1, 6)]
            if split == "train"
            else ["test_batch"]
        )

        for file in files:
            with open(os.path.join(base_dir, "cifar-10-batches-py", file), "rb") as f:
                data = pickle.load(f, encoding="bytes")

            images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = data[b"labels"]

            for i, (img, label) in enumerate(zip(images, labels)):
                if split == "train":
                    file_split = file.split("_")[-1]
                else:
                    file_split = split
                filename = f"{label_names[label]}_{file_split}_split_{i:06d}.png"
                path = os.path.join(img_dir, filename)
                Image.fromarray(img).save(path)
                all_data.append(
                    {
                        "fpath": path,
                        "file_name": filename,
                        "target": label,
                        "split": "train" if split == "train" else "test",
                    }
                )
    full_df = pd.DataFrame(all_data)
    train_df = full_df[full_df["split"] == "train"].drop(columns=["split"])
    test_df = full_df[full_df["split"] == "test"].drop(columns=["split"])
    return train_df, test_df
