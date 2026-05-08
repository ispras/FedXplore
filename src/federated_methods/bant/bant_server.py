import torch
import copy
from collections import OrderedDict
from hydra.utils import instantiate

from ..byzantine_base.byzantine_server import ByzantineBaseServer
from utils.losses import get_loss


class BANTServer(ByzantineBaseServer):
    def __init__(self, cfg, trust_df):
        super().__init__(cfg, trust_df)
        self.trust_df = trust_df
        self.client_model = instantiate(cfg.model, num_classes=trust_df.num_classes)

    def get_client_weights(self, client_gradients):
        client_weights = OrderedDict()
        for key, weight in self.initial_global_model_state.items():
            client_weights[key] = client_gradients[key].to(self.device) + weight
        return client_weights

    def get_trust_criterion(self):
        trust_criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.trust_df.data,
            num_classes=self.trust_df.num_classes,
        )
        return trust_criterion

    def get_trust_losses(self):
        trust_losses = []

        # Reinit loader to run from trainer
        self.prev_test_loader = self.test_loader
        self.test_loader = self.trust_loader

        # Reinit criterion to run from trainer
        tmp_criterion = self.criterion
        self.criterion = self.get_trust_criterion()

        self.initial_global_model_state = copy.deepcopy(self.global_model).state_dict()

        _, _, server_loss = self.model_trainer.server_eval_fn(self)
        print(f"Server trust loss: {server_loss}\n")
        for rank in self.list_clients:
            client_weights = self.get_client_weights(self.client_gradients[rank])
            self.global_model.load_state_dict(client_weights)
            _, _, client_loss = self.model_trainer.server_eval_fn(self)
            print(f"Client {rank} trust loss: {client_loss}")
            trust_losses.append(client_loss.cpu())

        # Revert attributes back
        self.global_model.load_state_dict(self.initial_global_model_state)
        self.criterion = tmp_criterion
        self.test_loader = self.prev_test_loader

        return server_loss.cpu(), trust_losses
