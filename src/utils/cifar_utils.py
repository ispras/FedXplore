import os
import yaml
import pickle
import tarfile
import requests
import pandas as pd
from PIL import Image


# Download


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

    return pd.DataFrame(all_data)


def set_data_configs(target_path):
    print("Setting paths to .yaml files...\n")
    # HARDCODE paths
    config_dir = "src/configs/dataset/"
    if not os.path.isdir(config_dir):
        print(
            f"Directory {config_dir} not found. Set paths inside .yaml configs manually"
        )
        return

    config_names = ["cifar10.yaml"]

    if not os.path.isabs(target_path):
        curent_run_path = os.getcwd()
        target_path = os.path.join(curent_run_path, target_path)

    for filename in os.listdir(config_dir):
        if filename not in config_names:
            continue

        filepath = os.path.join(config_dir, filename)
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        data_sources = data.get("data_sources", {})

        if "test_map_file" in data_sources:
            test_map_path = [os.path.join(target_path, "images", "test_map_file.csv")]
            data_sources["test_map_file"] = test_map_path

        if "train_map_file" in data_sources:
            train_map_name = "train_map_file.csv"
            train_map_path = [os.path.join(target_path, "images", train_map_name)]
            data_sources["train_map_file"] = train_map_path

        data["data_sources"] = data_sources

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
