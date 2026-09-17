import json
import os
import subprocess
import sys
import tempfile
import warnings

import mlflow
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils.utils import (
    get_run_command,
    get_repository_info,
    generate_confluence_report,
)


def redirect_stdout_to_log(run_dir):
    # Read output file (created by >output/file.txt)
    redirect_file = subprocess.run(
        ["readlink", "-f", f"/proc/{os.getpid()}/fd/1"], capture_output=True, text=True
    ).stdout
    redirect_file = redirect_file[: len(redirect_file) - 1]  # delete \n
    os.remove(redirect_file)

    # Get file to log learning (in dir created by hydra)
    absolute_run_dir = os.path.abspath(run_dir)
    main_log_file = os.path.join(absolute_run_dir, "output.txt")

    # Swap stdout to log file
    f = open(main_log_file, "w")
    sys.stdout = f
    sys.stderr = f

    # Create link to log file (output/file.txt link to log file)
    os.symlink(main_log_file, redirect_file)

    print("Information about files:")
    print(f"File to logging: {main_log_file}")
    print(f"Link file: {redirect_file}")
    return redirect_file


def build_client_participation_histogram(
    selection_df,
    num_clients,
    save_path,
    client_attack_map=None,
):
    all_clients = np.concatenate(selection_df["clients"].to_numpy())
    freq = np.bincount(all_clients, minlength=num_clients)
    client_ids = np.arange(num_clients)
    attack_types = [
        (client_attack_map or {}).get(client_id, "no_attack")
        for client_id in client_ids
    ]
    is_attacker = [attack_type != "no_attack" for attack_type in attack_types]

    summary = pd.DataFrame(
        {
            "client_id": client_ids,
            "rounds_selected": freq,
            "is_attacker": is_attacker,
            "attack_type": attack_types,
        }
    )

    benign_color = "tab:blue"
    attacker_color = "tab:red"
    colors = [
        attacker_color if attacked else benign_color
        for attacked in is_attacker
    ]

    plt.figure(figsize=(10, 4))
    plt.bar(client_ids, freq, color=colors)
    plt.xticks(client_ids)
    plt.xlabel("Client")
    plt.ylabel("Rounds selected")
    plt.title("Client participation frequency")
    if any(is_attacker):
        plt.legend(
            handles=[
                Patch(color=benign_color, label="Benign clients"),
                Patch(color=attacker_color, label="Attacking clients"),
            ]
        )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return summary


class BaseLogger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.redirect_file = redirect_stdout_to_log(self.run_dir)
        self.run_command = get_run_command()
        print(f"Run command: {self.run_command}\n")

    def end_logging(self):
        self.generate_confluence_report()

    def generate_confluence_report(self):
        if self.checkpoint_path is not None:
            self.report_file = generate_confluence_report(
                self.run_dir, self.checkpoint_path
            )
        else:
            git_info = get_repository_info()
            self.report_file = generate_confluence_report(
                run_dir=self.run_dir, git_info=git_info, run_command=self.run_command
            )
        self.report_file.close()

    def log_run_info(self, cfg):
        pass

    def log_scalar(self, scalar, name, cur_round):
        pass

    def log_pandas(self, pandas, group_name, cur_round):
        pass

    def save_artifact(self, content, artifact_name):
        pass


class MLFlowLogger(BaseLogger):
    def __init__(
        self,
        run_dir,
        tracking_uri,
        experiment_name,
        run_name,
        tags=None,
    ):
        super().__init__(run_dir)
        self.tracking_uri = (
            ""
            if tracking_uri in {None, "", "null"}
            else str(tracking_uri).strip()
        )
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = {
            str(key): str(value) for key, value in dict(tags or {}).items()
        }
        self._pending_metrics = {}
        self._pending_metric_step = None
        self.init_mlflow()

    def init_mlflow(self):
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        self.tracking_uri = mlflow.get_tracking_uri()
        experiment = mlflow.set_experiment(self.experiment_name)
        if experiment is None:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
        self.experiment_id = getattr(experiment, "experiment_id", None)
        active_run = mlflow.active_run()
        if active_run is None:
            started_run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
            self.run_id = started_run.info.run_id
        else:
            self.run_id = active_run.info.run_id
        self.run_url = ""
        if self.experiment_id and self.tracking_uri.startswith(("http://", "https://")):
            self.run_url = (
                self.tracking_uri.rstrip("/")
                + f"/#/experiments/{self.experiment_id}/runs/{self.run_id}"
            )
        print(f"MLFLOW_RUN_ID={self.run_id}")
        if self.experiment_id:
            print(f"MLFLOW_EXPERIMENT_ID={self.experiment_id}")
        if self.run_url:
            print(f"MLFLOW_RUN_URL={self.run_url}")
        metadata = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "tracking_uri": self.tracking_uri,
            "run_url": self.run_url,
            "run_name": self.run_name,
        }
        with open(os.path.join(self.run_dir, "mlflow_run.json"), "w") as file:
            json.dump(metadata, file, indent=2)

    def log_run_info(self, cfg):
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")
        self.git_info = get_repository_info()
        mlflow.log_dict(self.git_info, "git_info.yaml")
        mlflow.log_text(self.run_command, "run_command.txt")

    def log_scalar(self, scalar, name, cur_round):
        if scalar is None:
            return

        try:
            value = float(scalar)
        except (TypeError, ValueError):
            warnings.warn(f"Cannot log scalar {name} with value {scalar}")
            return

        if self._pending_metric_step not in {None, cur_round}:
            self._flush_metrics()
        self._pending_metric_step = cur_round
        self._pending_metrics[name] = value

    def _flush_metrics(self):
        if not self._pending_metrics:
            return
        mlflow.log_metrics(self._pending_metrics, step=self._pending_metric_step)
        self._pending_metrics = {}
        self._pending_metric_step = None

    def log_pandas(self, pandas, group_name, cur_round):
        """
        Log pandas DataFrame to MLflow as a set of scalar metrics.

        Expected DataFrame structure:
            - index: metric names (e.g. Accuracy, ROC-AUC, f1-score)
            - columns: classes / tasks / single column (e.g. cifar)

        Args:
            pandas (pd.DataFrame): metrics dataframe
            group_name (str): metric group prefix (e.g. 'test/', 'val/', 'clients/client_0/')
            cur_round: current step (round of FL)
        """
        if pandas is None:
            return

        if not isinstance(pandas, pd.DataFrame):
            warnings.warn(f"log_pandas expects pd.DataFrame, got {type(pandas)}")
            return

        # Ensure group_name ends with '/'
        if group_name and not group_name.endswith("/"):
            group_name = group_name + "/"

        metrics = {}
        for row_name in pandas.index:
            for col_name in pandas.columns:
                value = pandas.loc[row_name, col_name]

                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    warnings.warn(
                        f"Cannot log value for {group_name}{row_name}/{col_name}: {value}"
                    )
                    continue

                metric_name = f"{group_name}{row_name}_{col_name}"
                metrics[metric_name] = value
        if self._pending_metric_step == cur_round:
            metrics = {**self._pending_metrics, **metrics}
            self._pending_metrics = {}
            self._pending_metric_step = None
        if metrics:
            mlflow.log_metrics(metrics, step=cur_round)

    def save_artifact(self, content, artifact_name):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, artifact_name)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            mode = "wb" if isinstance(content, bytes) else "w"
            with open(path, mode) as f:
                f.write(content)

            mlflow.log_artifact(path)

    def end_logging(self):
        self._flush_metrics()
        super().end_logging()
        mlflow.end_run()
