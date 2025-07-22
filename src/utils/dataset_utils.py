import os
import yaml


def get_target_dir(cfg, default_dir="cifar10"):
    target_dir = (
        cfg.test_dataset.get("download_path")
        or cfg.train_dataset.get("download_path")
        or default_dir
    )
    if not os.path.isabs(target_dir):
        target_dir = os.path.join(os.getcwd(), target_dir)
    print(f"Download path: {target_dir}")
    return target_dir


def save_map_files(train_df, test_df, target_dir):
    train_df.to_csv(os.path.join(target_dir, "train_map_file.csv"), index=False)
    test_df.to_csv(os.path.join(target_dir, "test_map_file.csv"), index=False)


def set_data_configs(target_path, config_names=["cifar10.yaml"]):
    print("Setting paths to .yaml files...\n")
    # HARDCODE paths
    config_dir = "src/configs/dataset/"
    if not os.path.isdir(config_dir):
        print(
            f"Directory {config_dir} not found. Set paths inside .yaml configs manually"
        )
        return

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
            test_map_path = [os.path.join(target_path, "test_map_file.csv")]
            data_sources["test_map_file"] = test_map_path

        if "train_map_file" in data_sources:
            train_map_path = [os.path.join(target_path, "train_map_file.csv")]
            data_sources["train_map_file"] = train_map_path

        data["data_sources"] = data_sources

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
