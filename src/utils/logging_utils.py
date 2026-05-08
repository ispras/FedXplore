import os
import sys
import torch
import mlflow
import tempfile
import warnings
import subprocess
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

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


def build_client_participation_histogram(selection_df, num_clients, save_path):
    all_clients = np.concatenate(selection_df["clients"].to_numpy())

    freq = np.bincount(all_clients, minlength=num_clients)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(num_clients), freq)
    plt.xlabel("Client")
    plt.ylabel("Rounds selected")
    plt.title("Client participation frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
    ):
        super().__init__(run_dir)
        self.tracking_uri = str(tracking_uri)
        self.experiment_name = experiment_name
        self.run_name = run_name
        if self.tracking_uri.startswith("http://"):
            assert (
                self.tracking_uri == "http://10.100.202.109:5000/"
            ), f"We support only http://10.100.202.109:5000/ remote storage location, you provide: {self.tracking_uri}"
            if not self.aws_credentials_present():
                warnings.warn("aws credentials not set up. We set it manually")
                self.set_up_aws_credentials()
        self.init_mlflow()

    def init_mlflow(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        active_run = mlflow.active_run()
        if active_run is None:
            started_run = mlflow.start_run(run_name=self.run_name)
            self.run_id = started_run.info.run_id
        else:
            self.run_id = active_run.info.run_id

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

        mlflow.log_metric(name, value, step=cur_round)

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
                mlflow.log_metric(metric_name, value, step=cur_round)

    def save_artifact(self, content, artifact_name):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, artifact_name)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            mode = "wb" if isinstance(content, bytes) else "w"
            with open(path, mode) as f:
                f.write(content)

            mlflow.log_artifact(path)

    def end_logging(self):
        super().end_logging()
        # Update current best model state with run_id info
        if hasattr(self, "state") and self.state is not None:
            OmegaConf.update(self.state["config_file"], "logger.run_id", self.run_id)
            # Save model info
            torch.save(self.state, self.checkpoint_path)
        mlflow.end_run()

    def aws_credentials_present(self):
        env_ok = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
            and os.environ.get("AWS_ENDPOINT_URL")
        )
        cred_file = os.path.exists(
            os.path.expanduser("~/.aws/credentials")
        ) or os.path.exists(os.path.expanduser("~/.aws/config"))
        return env_ok or cred_file

    def set_up_aws_credentials(self):
        os.environ["AWS_ACCESS_KEY_ID"] = "ecgadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "d3po6901"
        os.environ["AWS_ENDPOINT_URL"] = "http://10.100.151.14:9000"
