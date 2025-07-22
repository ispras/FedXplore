import copy
import time
from collections import OrderedDict
from hydra.utils import instantiate

from ..byzantine_base.byzantine_server import ByzantineBaseServer
from utils.losses import get_loss


class FLTrustServer(ByzantineBaseServer):
    def __init__(self, cfg, fltrust_df):
        super().__init__(cfg, fltrust_df)
        self.initial_global_model_state = None
        self.server_grad = OrderedDict()

    def _init_optimizer(self):
        self.optimizer = instantiate(
            self.cfg.optimizer, params=self.global_model.parameters()
        )

    def _init_criterion(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.trust_df,
        )

    def get_loss_value(self, outputs, targets):
        return self.criterion(outputs, targets)

    def train_fn(self):
        self.global_model.to(self.device)
        self.global_model.train()
        for _ in range(self.cfg.federated_params.local_epochs):

            for batch in self.trust_loader:
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.global_model(inp)

                loss = self.get_loss_value(outputs, targets)

                loss.backward()

                self.optimizer.step()

    def get_grad(self):
        self.global_model.eval()
        for key, _ in self.global_model.state_dict().items():
            self.server_grad[key] = self.global_model.state_dict()[key].to(
                "cpu"
            ) - self.initial_global_model_state[key].to("cpu")

    def fltrust_train(self):
        print(f"\nServer started training")
        start = time.time()

        self._init_optimizer()
        self._init_criterion()

        self.initial_global_model_state = copy.deepcopy(self.global_model).state_dict()

        # Train server
        self.train_fn()
        # For now grad is the diff between trained model and server state
        self.get_grad()
        print(f"Server finished training in {time.time() - start} sec")
        # Return the initial state
        self.global_model.load_state_dict(self.initial_global_model_state)
