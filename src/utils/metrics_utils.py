import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

__all__ = [
    "stopping_criterion",
]


def calculate_cifar_metrics(fin_targets, results, verbose=False):
    df = pd.DataFrame(
        columns=["cifar"],
        index=[
            "Accuracy",
            "Precision",
            "Recall",
            "f1-score",
        ],
    )
    df.loc["Accuracy", "cifar"] = accuracy_score(fin_targets, results)
    df.loc["Precision", "cifar"] = precision_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["Recall", "cifar"] = recall_score(
        fin_targets, results, average="macro", zero_division=0
    )
    df.loc["f1-score", "cifar"] = f1_score(
        fin_targets, results, average="macro", zero_division=0
    )
    if verbose:
        print(df)
    return df


def stopping_criterion(
    val_loss,
    metrics,
    best_metrics,
    rounds_no_improve,
):
    """
    Define stopping criterion for metrics from config['saving_metrics']
    best_metrics is updated only if every metric from best_metrics.keys() has improved

    :param val_loss: validation loss
    :param metrics: validation metrics
    :param best_metrics: the best metrics for the current round
    :param rounds_no_improve: number of rounds without best_metrics updating

    :return: rounds_no_improve, best_metrics
    """
    # get average metrics by class
    metrics = dict(metrics.mean(axis=1))

    # define condition best_metric >= metric for all except for loss
    metrics_mask = all(
        metrics[key] >= best_metrics[key] for key in best_metrics.keys() - {"loss"}
    )
    if not metrics_mask:
        rounds_no_improve += 1
        return rounds_no_improve, best_metrics
    if "loss" in list(best_metrics.keys()):
        if val_loss >= best_metrics["loss"]:
            rounds_no_improve += 1
            return rounds_no_improve, best_metrics

    # Updating best_metrics
    for key in list(best_metrics.keys()):
        if key == "loss":
            best_metrics[key] = val_loss
        else:
            best_metrics[key] = metrics[key]

    # Updating rounds_no_improve
    rounds_no_improve = 0
    return rounds_no_improve, best_metrics


def check_metrics_names(metrics):
    allowed_metrics = [
        "loss",
        "Specificity",
        "Sensitivity",
        "G-mean",
        "f1-score",
        "fbeta2-score",
        "ROC-AUC",
        "AP",
        "Precision (PPV)",
        "NPV",
    ]

    assert all(
        [k in allowed_metrics for k in metrics.keys()]
    ), f"federated_params.server_saving_metrics can be only {allowed_metrics}, but get {metrics.keys()}"
