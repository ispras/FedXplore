import pandas as pd
from torch.utils.data import DataLoader


def get_dataset_loader(
    dataset,
    cfg,
    drop_last=True,
):
    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=(dataset.mode == "train"),
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    assert (
        len(loader) > 0
    ), f"len(dataloader) is 0, either lower the batch size, or put drop_last=False"
    return loader


def print_df_distribution(df, num_classes, num_clients, pathology_names=None):
    is_multilabel = isinstance(df["target"].iloc[0], list)
    if is_multilabel:
        df["class_target"] = df["target"].apply(
            lambda x: [i for i, val in enumerate(x) if val == 1] or [-1]
        )

    print(f"Total usage data: {len(df[df['client'] != -1])}")

    client_groups = df.groupby("client")
    valid_clients = set(df["client"].unique())

    for cl in range(num_clients):
        if cl not in valid_clients:
            print(f"Client {cl:>2} | No data")
            continue

        client_data = client_groups.get_group(cl)
        if is_multilabel:
            distr = (
                client_data["class_target"]
                .explode()
                .value_counts()
                .reindex(range(-1, num_classes), fill_value=0)
                .tolist()
            )
        else:
            distr = (
                client_data["target"]
                .value_counts()
                .reindex(range(num_classes), fill_value=0)
                .tolist()
            )
        x_str = " ".join(f"{num:>5}" for num in distr)
        if cl == 0 and is_multilabel:
            pathology_str = " " * 11 + "  ".join(["Other"] + pathology_names)
            print(pathology_str)
        print(f"Client {cl:>3} | {x_str} | {len(client_data):>5}")


def create_distribution_md(
    dataset: pd.DataFrame,
    num_classes: int,
    num_clients: int,
):
    """
    Create markdown table with client data distribution.

    Args:
        df (pd.DataFrame): dataset with columns ['client', 'target']
        num_classes (int): number of classes
        num_clients (int): total number of clients

    Returns:
        str: markdown-formatted table
    """
    pathology_names = (
        dataset.dataset_cfg.pathology_names
        if (
            hasattr(dataset, "dataset_cfg")
            and hasattr(dataset.dataset_cfg, "pathology_names")
        )
        else None
    )
    df = dataset.data
    is_multilabel = isinstance(df["target"].iloc[0], list)

    work_df = df.copy()

    if is_multilabel:
        work_df["class_target"] = work_df["target"].apply(
            lambda x: [i for i, val in enumerate(x) if val == 1] or [-1]
        )

    total_samples = len(work_df[work_df["client"] != -1])

    rows = []

    valid_clients = set(work_df["client"].unique())
    client_groups = work_df.groupby("client")

    for cl in range(num_clients):
        if cl not in valid_clients:
            distr = [0] * num_classes
            total = 0
        else:
            client_data = client_groups.get_group(cl)

            if is_multilabel:
                counts = (
                    client_data["class_target"]
                    .explode()
                    .value_counts()
                    .reindex(range(-1, num_classes), fill_value=0)
                )
                # drop "-1" (Other) if pathology_names provided
                if pathology_names is not None:
                    counts = counts.drop(-1)
                distr = counts.tolist()
            else:
                distr = (
                    client_data["target"]
                    .value_counts()
                    .reindex(range(num_classes), fill_value=0)
                    .tolist()
                )

            total = len(client_data)

        row = {"Client": f"Client {cl}"}
        for i, val in enumerate(distr):
            col_name = pathology_names[i] if pathology_names is not None else f"{i}"
            row[col_name] = val

        row["Total"] = total
        rows.append(row)

    df_md = pd.DataFrame(rows)

    md = []
    md.append(f"### Client data distribution\n")
    md.append(f"**Total usage data:** {total_samples}\n")
    md.append(df_md.to_markdown(index=False))

    return "\n".join(md)
