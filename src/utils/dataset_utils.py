import os
from pathlib import Path

try:
    import zarr
except ImportError:  # pragma: no cover - optional dependency
    zarr = None
try:
    from ecglib.data.datasets import EcgDataset
except ImportError:  # pragma: no cover - optional dependency
    EcgDataset = None


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


def resolve_data_source_path(base_path, entry):
    candidate = Path(str(entry)).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    root = Path(str(base_path)).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return str((root / candidate).resolve())


def update_data_sources(base_path, data_sources):
    # Normalize relative data-source paths against `base_path`.
    # Absolute paths are preserved as-is.
    for data_sources_type in list(data_sources.keys()):
        entries = data_sources[data_sources_type]
        if entries is None:
            continue
        if isinstance(entries, str):
            entries = [entries]
        data_sources[data_sources_type] = [
            resolve_data_source_path(base_path, entry) for entry in entries
        ]

    return data_sources


class ZarrEcgDataset(EcgDataset if EcgDataset is not None else object):
    """
    Custom ecglib.data.dataset.EcgDataset class to support filesystem-backed zarr ECG data.

    1. Load zarr.Directory store
    2. Overwrite `read_ecg_record` to zarr.load
    """

    def __init__(
        self,
        ecg_data,
        target,
        frequency=500,
        leads=...,
        data_type="wfdb",
        ecg_length=10,
        cut_range=...,
        pad_mode="constant",
        norm_type="z_norm",
        classes=2,
        augmentation=None,
        path_to_zarr=None,
    ):
        if EcgDataset is None:
            raise RuntimeError(
                "ecglib is required to use ZarrEcgDataset. Install it in your environment first."
            )
        super().__init__(
            ecg_data,
            target,
            frequency,
            leads,
            data_type,
            ecg_length,
            cut_range,
            pad_mode,
            norm_type,
            classes,
            augmentation,
        )
        assert (
            self.data_type == "zarr"
        ), f"You can't use ZarrEcgDataset if `data_type` is not 'zarr', you provide: {self.data_type}"
        if zarr is None:
            raise RuntimeError(
                "zarr is required to use ZarrEcgDataset. Install it in your environment first."
            )
        store = zarr.DirectoryStore(path_to_zarr)
        self.root = zarr.open_group(store, mode="r")

    def read_ecg_record(self, file_path, data_type):
        ecg_record = self.root[file_path][...].astype("float64")
        return ecg_record
