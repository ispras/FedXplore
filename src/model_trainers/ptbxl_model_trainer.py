import torch
import pandas as pd
import numpy as np
import math
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
    fbeta_score,
)
from hydra.utils import instantiate
import warnings
from utils.data_utils import get_dataset_loader

warnings.filterwarnings("ignore")


class PTBXLModelTrainer:
    def __init__(self, cfg, metrics_for_threshold):
        # Always remember that you most likely
        # won't be able to change any variables in `context`,
        # since python is passed a copy of this object.
        # The exception would be mutable fields in the `context`.
        self.metrics_for_threshold = metrics_for_threshold
        self.cfg = cfg

        self.trust_dataset = None
        self.trust_loader = None

        if "trust_dataset" in self.cfg:
            self.trust_dataset = instantiate(
                self.cfg.trust_dataset, cfg=cfg, mode="trust", _recursive_=False
            )
            self.trust_loader = get_dataset_loader(
                self.trust_dataset, cfg, drop_last=False
            )
        else:
            warnings.warn(
                """Trust dataset is not specified, 
                so there will be no selection of the threshold on the server. 
                It will be set to 0.5 for testing."""
            )

    def train_fn(self, context):
        # context.model must be a torch.nn.Module,
        # or other mutable object, for this function to work

        context.model.train()
        for _ in range(context.local_epochs):
            for batch in context.train_loader:
                _, (input, targets) = batch

                inp = input[0].to(context.device)
                targets = targets.to(context.device)

                context.optimizer.zero_grad()
                outputs = context.model(inp)

                loss = context.get_loss_value(outputs, targets)
                loss.backward()

                context.optimizer.step()

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

    def client_eval_fn(self, context):
        context.model.eval()
        val_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(context.valid_loader):
                _, (input, targets) = batch

                inp = input[0].to(context.device)
                targets = targets.to(context.device)

                outputs = context.model(inp)

                val_loss += context.criterion(outputs, targets).detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

        client_metrics = self.calculate_metrics(fin_targets, fin_outputs, context)
        return val_loss / len(context.valid_loader), client_metrics

    def test_fn(self, context):
        fin_targets, fin_outputs, test_loss = self.server_eval_fn(context)
        metrics = self.calculate_metrics(
            fin_targets,
            fin_outputs,
            context,
            verbose=True,
        )
        return metrics, test_loss

    def server_eval_fn(self, context):
        context.global_model.to(context.device)
        context.global_model.eval()

        test_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(context.test_loader):
                _, (input, targets) = batch

                inp = input[0].to(context.device)
                targets = targets.to(context.device)
                outputs = context.global_model(inp)

                test_loss += context.criterion(outputs, targets)
                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        test_loss /= len(context.test_loader)
        return fin_targets, fin_outputs, test_loss

    def calculate_metrics(self, fin_targets, fin_outputs, context, verbose=False):
        # Assume that train, test have the same pathologies
        # Therefore, we select pathologies only by train
        pathology_names = (
            list(context.cfg.train_dataset.dataset_cfg.merge_map.keys())
            if context.cfg.train_dataset.dataset_cfg.merge_map
            else context.cfg.train_dataset.dataset_cfg.pathology_names
        )

        # Get results
        sigmoid = torch.nn.Sigmoid()
        fin_outputs = sigmoid(torch.as_tensor(fin_outputs))

        # We find the prediction threshold differently for the client and server
        # Since selecting a treshhold based on a test data is unacceptable
        prediction_threshold = 0.5
        if hasattr(context, "rank"):
            # We will select a threshold for the validation data
            # on the client using current validation results
            prediction_threshold = select_best_validation_threshold(
                fin_targets, fin_outputs, self.metrics_for_threshold
            )
        elif hasattr(context, "global_model") and (self.trust_dataset is not None):
            # We will select a threshold for the test data
            # on the server side by using a trust dataset

            # Replace the server loader with a trust loader
            server_test_loader = context.test_loader
            context.test_loader = self.trust_loader

            # Eval global model on trust dataset
            trust_targets, trust_outputs, trust_loss = self.server_eval_fn(context)
            trust_outputs = sigmoid(torch.as_tensor(trust_outputs))

            # Select best threshold on trust data
            prediction_threshold = select_best_validation_threshold(
                trust_targets, trust_outputs, self.metrics_for_threshold
            )
            context.test_loader = server_test_loader

        results = (fin_outputs > prediction_threshold).float().tolist()

        # Calc metrics
        metrics, conf_matrix = compute_metrics(
            fin_targets, results, pathology_names, fin_outputs
        )

        if verbose:
            print(metrics)
            print(conf_matrix)
            print(classification_report(fin_targets, results, zero_division=False))

        return metrics


def select_best_validation_threshold(
    fin_targets,
    fin_outputs,
    metrics_threshold,
):
    assert all(k in ["gmean", "f1-score", "ROC-AUC"] for k in metrics_threshold.keys())
    assert sum(v for v in metrics_threshold.values()) == 1.0

    fin_targets = torch.tensor(fin_targets)
    fin_outputs = torch.tensor(fin_outputs.clone().detach())
    thresholds = torch.arange(-0.01, 0.9, 0.06)
    M = fin_targets.size(1)  # Number of classes
    best_thresholds = []

    for class_idx in range(M):
        # Get target and output for the current class
        class_targets = fin_targets[:, class_idx]
        class_outputs = fin_outputs[:, class_idx]

        # Compute true positives, false positives, and positive outputs for all thresholds
        bool_outputs = (class_outputs.unsqueeze(1) > thresholds).float()  # [N, T]
        positive_outputs = bool_outputs.sum(
            dim=0
        )  # Sum over samples for each threshold

        tp = (bool_outputs * class_targets.unsqueeze(1)).sum(dim=0)  # True positives
        fp = (bool_outputs * (1 - class_targets.unsqueeze(1))).sum(
            dim=0
        )  # False positives
        positive_samples = class_targets.sum().item()
        negative_samples = len(class_targets) - positive_samples

        tpr = tp / (positive_samples + 1e-10)  # True positive rate
        fpr = fp / (negative_samples + 1e-10)  # False positive rate
        precision = tp / (positive_outputs + 1e-10)  # Precision

        # Compute metrics
        metrics = {}

        if "gmean" in metrics_threshold:
            metrics["gmean"] = torch.sqrt(tpr * (1 - fpr))
        if "f1-score" in metrics_threshold:
            metrics["f1-score"] = 2 * (tpr * precision) / (tpr + precision + 1e-10)
        if "ROC-AUC" in metrics_threshold:
            metrics["ROC-AUC"] = (tpr + (1 - fpr)) / 2

        # Calculate composite metric
        composite_metric = sum(
            metrics_threshold[k] * metrics[k] for k in metrics_threshold.keys()
        )

        # Get the best threshold
        best_threshold_idx = np.argmax(composite_metric)
        best_thresholds.append(thresholds[best_threshold_idx].item())

    return torch.tensor(best_thresholds)


def compute_metrics(
    target,
    prediction,
    pathology_names,
    probs,
):
    """
    Compute metrics from input predictions and
    ground truth labels

    :param target: Ground truth labels
    :param prediction: Predicted labels
    :param pathology_names: Class names of pathologies
    :return: pandas DataFrame with metrics and confusion matrices for each class
    """
    df = pd.DataFrame(
        columns=pathology_names,
        index=[
            "Specificity",
            "Sensitivity",
            "G-mean",
            "f1-score",
            "fbeta2-score",
            "ROC-AUC",
            "AP",
            "Precision_PPV_",
            "NPV",
        ],
    )
    conf_mat_df = pd.DataFrame(columns=["TN", "FP", "FN", "TP"], index=pathology_names)
    target = np.array(target, int)
    prediction = np.array(prediction, int)
    probs = np.array(probs)

    for i, col in enumerate(pathology_names):
        tn, fp, fn, tp = get_confusion_matrix(target[:, i], prediction[:, i])
        conf_mat_df.loc[col] = [tn, fp, fn, tp]

        spec = divide(tn, tn + fp)
        sens = divide(tp, tp + fn)

        df.loc["Specificity", col] = spec
        df.loc["Sensitivity", col] = sens
        df.loc["G-mean", col] = math.sqrt(spec * sens)
        df.loc["Precision_PPV_", col] = divide(tp, tp + fp)
        df.loc["NPV", col] = divide(tn, tn + fn)
        df.loc["f1-score", col] = f1_score(
            target[:, i], prediction[:, i], zero_division=0
        )
        df.loc["fbeta2-score", col] = fbeta_score(
            target[:, i], prediction[:, i], beta=2, zero_division=0
        )
        try:
            df.loc["ROC-AUC", col] = roc_auc_score(target[:, i], probs[:, i])
        except:
            df.loc["ROC-AUC", col] = np.nan
        try:
            df.loc["AP", col] = average_precision_score(target[:, i], prediction[:, i])
        except:
            df.loc["AP", col] = np.nan

    df = df.astype(float).round(3)
    return df, conf_mat_df


def get_confusion_matrix(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return tn, fp, fn, tp


def divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan
