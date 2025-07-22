import torch
import torch.nn as nn
import numpy as np

from ..base.base_server import BaseServer
from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from .autobant_models import AutoBANTModel2d, AutoBANTModel1d


class AutoBANTServer(BaseServer):
    def __init__(
        self,
        cfg,
        trust_df,
        start_point,
        end_point,
        num_opt_epochs,
        mirror_gamma,
        ts_momentum,
    ):
        super().__init__(cfg)
        self.trust_df = trust_df
        self.trust_loader = get_dataset_loader(self.trust_df, cfg, drop_last=False)
        assert start_point in [
            "uniform",
            "previous",
            "best",
        ], f"start point in trust scores optimization can be one of ['uniform', 'previous', 'best'], you provide: {start_point}"
        assert end_point in [
            "origin"
        ], f"end point in trust scores optimization can be one of ['origin'], you provide: {end_point}"
        self.start_point = start_point
        self.end_point = end_point
        self.num_opt_epochs = num_opt_epochs
        self.mirror_gamma = mirror_gamma
        self.ts_momentum = ts_momentum
        self.num_clients_subset = self.cfg.federated_params.client_subset_size
        self.start_trust_scores = torch.tensor(
            [1 / self.num_clients_subset] * self.num_clients_subset
        )

    def _init_criterion(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.trust_df,
        )

    def _init_trust_model(self):
        # HARDCODE
        # At the moment the method supports only 2 models: resnet1d18, resnet18
        if "resnet1d18" in self.cfg.model["_target_"]:
            model_class = AutoBANTModel1d
        elif "resnet18" in self.cfg.model["_target_"]:
            model_class = AutoBANTModel2d
        else:
            raise ValueError(
                "At the moment the method supports only 2 models: resnet1d18, resnet18"
            )

        self.trust_model = model_class(
            self.cfg,
            self.global_model.state_dict(),
            [self.client_gradients[rank] for rank in self.list_clients],
            self.device,
            self.start_trust_scores,
        )

    def set_trust_model_weights(self, trust_scores):
        self.trust_model.trust_scores = nn.Parameter(trust_scores, requires_grad=True)

    def _calc_mirror_step(self, trust_scores, trust_grads):
        numerator = trust_scores * torch.exp(-self.mirror_gamma * trust_grads)
        denominator = torch.sum(numerator)
        return numerator / denominator

    def _update_ts_with_momentum(self, trust_scores):
        return (
            1 - self.ts_momentum
        ) * self.trust_model.trust_scores + self.ts_momentum * trust_scores

    def _count_trust_score(self):
        best_trust_scores = self.start_trust_scores
        trust_scores = self.start_trust_scores
        best_train_loss = np.inf
        self._init_criterion()
        self._init_trust_model()
        # Solve optimization task
        for epoch in range(self.num_opt_epochs):
            epoch_loss = 0
            for batch in self.trust_loader:
                # Train step
                self.set_trust_model_weights(trust_scores)
                loss = self.calc_trust_loss(batch)
                # optimizer.step() on unit simplex
                trust_scores = self._calc_mirror_step(
                    self.trust_model.trust_scores,
                    self.trust_model.trust_scores.grad,
                )
                # add momentum
                trust_scores = self._update_ts_with_momentum(trust_scores)
                # update best trust scores
                if self.end_point == "origin" and loss < best_train_loss:
                    best_trust_scores = self.trust_model.trust_scores.clone().detach()
                    best_train_loss = loss
                epoch_loss += loss
            epoch_loss /= len(self.trust_loader) + int(
                bool(len(self.trust_df) % len(self.trust_loader))
            )
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Best Loss: {best_train_loss}")
        if self.start_point == "previous":
            self.start_trust_scores = trust_scores
        if self.start_point == "best":
            self.start_trust_scores = best_trust_scores
        if self.end_point == "origin":
            for i in range(self.num_clients_subset):
                if best_trust_scores[i] < 0.5 / self.num_clients_subset:
                    best_trust_scores[i] = 0
        return best_trust_scores

    def calc_trust_loss(self, batch):
        _, (input, targets) = batch

        inp = input[0].to(self.device)
        targets = targets.to(self.device)

        outputs = self.trust_model(inp)

        loss = self.criterion(outputs, targets)
        loss.backward()

        return loss
