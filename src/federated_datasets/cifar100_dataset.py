import os
import tarfile
import requests
import pickle
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from .federated_dataset import FederatedDataset
from utils.dataset_utils import set_data_configs, get_target_dir, save_map_files


class Cifar100Dataset(FederatedDataset):
    def __init__(self, cfg, mode, data_sources, base_path, **kwargs):
        # Set transform for CIFAR-100 dataset
        self.transform = self.set_up_transform(mode)
        super().__init__(cfg, mode, data_sources, base_path)

    def set_up_transform(self, mode):
        image_size = 32
        # CIFAR-100 statistics
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)
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
        print("We will download CIFAR-100 dataset!")
        print(
            "You can restart the experiment and set `+train_dataset.download_path` "
            "to specify the path to save the dataset."
        )

        # default target directory name (can be overridden via config)
        target_dir = get_target_dir(self.cfg, default_dir="cifar100")

        # 1. Download dataset
        download_cifar100(target_dir)
        # 2. Convert to images, create map-files
        train_df, test_df = process_cifar100(target_dir)
        # 3. Update instantiated Dataset class
        self.target_dir = os.path.join(target_dir, "images")
        super().downloading()
        # 4. Save map-files in target directory
        save_map_files(train_df, test_df, self.target_dir)
        # 5. Update paths in yaml config
        set_data_configs(self.target_dir, config_names=["cifar100.yaml"])

    def __getitem__(self, index):
        image = Image.open(self.data["fpath"][index])
        image = self.transform(image)
        label = self.data["target"][index]
        return index, ([image], label)


def download_cifar100(target_dir="cifar100"):
    os.makedirs(target_dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    tar_path = os.path.join(target_dir, "cifar-100-python.tar.gz")

    if not os.path.exists(tar_path):
        print("Downloading CIFAR-100...")
        response = requests.get(url, stream=True)
        with open(tar_path, "wb") as f:
            f.write(response.content)

    # Extract files
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    os.remove(tar_path)


def process_cifar100(base_dir="cifar100"):
    img_dir = os.path.join(base_dir, "images", "data")
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.isabs(img_dir):
        curent_run_path = os.getcwd()
        img_dir = os.path.join(curent_run_path, img_dir)

    # Load meta for label names (support both bytes and str keys)
    meta_path = os.path.join(base_dir, "cifar-100-python", "meta")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    label_names = [ln.decode("utf-8") for ln in meta[b"fine_label_names"]]

    all_data = []
    print("Converting CIFAR-100...", flush=True)
    for split in ["train", "test"]:
        file_path = os.path.join(base_dir, "cifar-100-python", split)
        with open(file_path, "rb") as f:
            try:
                data = pickle.load(f, encoding="bytes")
            except TypeError:
                f.seek(0)
                data = pickle.load(f)

        # data key retrieval (support bytes and str keys)
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected data format in CIFAR-100 {split} file.")
        data_key = b"data" if b"data" in data else "data"

        # fine (100-class) labels
        if b"fine_labels" in data:
            fine_labels = data[b"fine_labels"]
        elif "fine_labels" in data:
            fine_labels = data["fine_labels"]
        elif b"labels" in data:
            fine_labels = data[b"labels"]
        elif "labels" in data:
            fine_labels = data["labels"]
        else:
            raise RuntimeError(f"Cannot find fine labels in CIFAR-100 {split} file.")
        # coarse (20-class) labels
        if b"coarse_labels" in data:
            coarse_labels = data[b"coarse_labels"]
        elif "coarse_labels" in data:
            coarse_labels = data["coarse_labels"]
        else:
            raise RuntimeError(f"Cannot find coarse labels in CIFAR-100 {split} file.")

        images = data[data_key].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        for i, (img, fine_y, coarse_y) in enumerate(
            zip(images, fine_labels, coarse_labels)
        ):
            filename = f"{label_names[fine_y]}_{split}_split_{i:06d}.png"
            path = os.path.join(img_dir, filename)
            Image.fromarray(img).save(path)
            all_data.append(
                {
                    "fpath": path,
                    "file_name": filename,
                    "target": int(fine_y),
                    "coarse_target": int(coarse_y),
                    "split": "train" if split == "train" else "test",
                }
            )

    full_df = pd.DataFrame(all_data)
    train_df = full_df[full_df["split"] == "train"].drop(columns=["split"])
    test_df = full_df[full_df["split"] == "test"].drop(columns=["split"])
    return train_df, test_df
