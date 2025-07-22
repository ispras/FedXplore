import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from hydra.utils import instantiate

from utils.utils import create_model_info
from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from utils.metrics_utils import (
    stopping_criterion,
    check_metrics_names,
)


class BaseServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client_gradients = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.server_metrics = [
            pd.DataFrame() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.test_df = instantiate(
            cfg.test_dataset, cfg=cfg, mode="test", _recursive_=False
        )
        self.test_loader = get_dataset_loader(self.test_df, cfg, drop_last=False)
        self.device = (
            "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.test_df,
        )
        self.model_trainer = instantiate(
            self.cfg.model_trainer, cfg=self.cfg, _recursive_=False
        )
        self.model_path = self.create_model_path()
        self.best_metrics = {
            metric: 1000 * (metric == "loss")
            for metric in cfg.federated_params.server_saving_metrics
        }
        check_metrics_names(self.best_metrics)
        self.metric_aggregation = cfg.federated_params.server_saving_agg
        assert self.metric_aggregation in [
            "uniform",
            "weighted",
        ], f"federated_params.server_saving_agg can be only ['uniform', 'weighted'], you provide: {self.best_metrics}"
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        self.best_round = 0
        self.rounds_no_improve = 0
        self.last_metrics = None
        self.global_model = None
        self.cur_round = None
        self.list_clients = None

    def test_global_model(self):
        print(f"\nServer Test Results:")
        self.last_metrics, self.test_loss = self.model_trainer.test_fn(self)
        print(f"Server Test Loss: {self.test_loss}")

    def select_clients_to_train(self, subsample_amount):
        raise NotImplementedError(
            "Base does not have a client selection function, it will be changed by Client Selector"
        )

    def set_client_result(self, client_result):
        # Put client information in accordance with his rank
        self.client_gradients[client_result["rank"]] = client_result["grad"]
        self.server_metrics[client_result["rank"]] = client_result["server_metrics"]

    def save_best_model(self, round):
        # Collect metrics from clients
        # server_metrics = (metrics, val_loss, len(val_df))
        server_metrics = [
            metrics[0] for metrics in self.server_metrics if len(metrics) != 0
        ]
        val_losses = [
            metrics[1] for metrics in self.server_metrics if len(metrics) != 0
        ]
        val_len_dfs = [
            metrics[2] for metrics in self.server_metrics if len(metrics) != 0
        ]

        weights = [val_len_df / sum(val_len_dfs) for val_len_df in val_len_dfs]
        metrics_names = server_metrics[0].index

        if self.metric_aggregation == "uniform":
            # Uniform metrics agregation
            val_loss = np.mean(val_losses)
            metrics = pd.concat(server_metrics).groupby(level=0).mean()
        if self.metric_aggregation == "weighted":
            # Weighted metrics aggregation
            val_loss = np.sum(
                [loss * weight for loss, weight in zip(val_losses, weights)]
            )
            metrics = sum(
                weight * metric for weight, metric in zip(weights, server_metrics)
            )
        metrics = metrics.reindex(metrics_names)
        print(f"\nServer Valid Results:\n{metrics}")
        print(f"Server Valid Loss: {val_loss}")
        # Update best metrics
        rounds_no_improve, best_metrics = stopping_criterion(
            val_loss,
            metrics,
            self.best_metrics,
            rounds_no_improve=self.rounds_no_improve,
        )
        if rounds_no_improve == 0 and val_loss is not np.nan:
            print("\nServer model saved!")
            prev_model_path = f"{self.model_path}_round_{self.best_round}.pt"
            if os.path.exists(prev_model_path):
                os.remove(prev_model_path)
            self.best_metrics = best_metrics
            self.best_round = round
            checkpoint_path = f"{self.model_path}_round_{self.best_round}.pt"
            model_info = create_model_info(
                model_state=self.global_model.state_dict(),
                metrics=self.last_metrics,
                cfg=self.cfg,
            )
            torch.save(model_info, checkpoint_path)
        # Print comparing results
        metrics.loc["loss"] = val_loss
        print(f"\nCriterion metrics:")
        for k, v in self.best_metrics.items():
            print(
                f"Current {k}: {metrics.loc[k].mean()}\nBest {k}: {v}\nBest round: {self.best_round}\n",
            )

    def create_model_path(self):
        self.target_label_names = [self.test_df.name]

        return f"{self.cfg.single_run_dir}/{type(instantiate(self.cfg.federated_method, _recursive_=False)).__name__}_{'_'.join(self.target_label_names)}"

    def send_content_to_client(self, pipe_num, content):
        # Send content to client
        self.pipes[pipe_num].send(content)

    def rcv_content_from_client(self, pipe_num):
        # Get content from current client
        client_content = self.pipes[pipe_num].recv()

        return client_content
