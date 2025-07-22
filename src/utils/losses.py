import torch
import numpy as np


def get_loss(loss_cfg, df=None, device=None, init_pos_weight=None):
    loss_name = loss_cfg.loss_name
    loss = loss_cfg.config
    if loss_name == "ce":
        if init_pos_weight:
            pos_weight = calculate_class_weights_multi_class(df)
            pos_weight = torch.tensor(pos_weight).to(device)
        else:
            pos_weight = None
        return torch.nn.CrossEntropyLoss(
            weight=pos_weight,
            ignore_index=loss.ignore_index,
            reduction=loss.reduction,
            label_smoothing=loss.label_smoothing,
        )
    elif loss_name == "bce":
        if init_pos_weight:
            pos_weight = calculate_pos_weight(df)
            pos_weight = torch.tensor([pos_weight]).to(device)
        elif loss.pos_weight is not None:
            pos_weight = loss.pos_weight
            pos_weight = torch.tensor([pos_weight]).to(device)
        else:
            pos_weight = None

        return torch.nn.BCEWithLogitsLoss(
            reduction=loss.reduction,
            pos_weight=pos_weight,
        )
    else:
        raise ValueError("Unknown type of loss function")


def calculate_pos_weight(df):
    # Convert the 'target' column into a NumPy array
    target_array = np.array(df["target"].tolist())

    # Count zeros and ones for each index
    zeros_count = np.sum(target_array == 0, axis=0)
    ones_count = np.sum(target_array == 1, axis=0)

    # Calculate portion (ratio) of zeros to ones for each index
    ratios = np.divide(
        zeros_count,
        ones_count,
        out=np.ones_like(zeros_count, dtype=float),
        where=(ones_count != 0),
    )
    return ratios.tolist()


def calculate_class_weights_multi_class(df):
    df_copy = df.copy()
    target_array = np.array(df_copy["target"].tolist())

    class_weights = {}
    ordered_weights = []

    unique_classes = np.unique(target_array)
    total_count = len(target_array)

    for cls in unique_classes:
        class_count = np.sum(target_array == cls, axis=0)
        class_weight = float(total_count / (len(unique_classes) * class_count))
        class_weights[cls] = class_weight

    # Ensure the weights are added to the list in order of ascending class index
    for cls in sorted(unique_classes):
        ordered_weights.append(class_weights[cls])

    return ordered_weights
