import os
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
