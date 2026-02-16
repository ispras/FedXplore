import torch
import copy
from itertools import islice

from ..byzantine_base.byzantine_server import ByzantineBaseServer


class ZenoServer(ByzantineBaseServer):
    def __init__(self, cfg, trust_df, use_buffers, ro, b):
        super().__init__(cfg, trust_df)
        self.use_buffers = use_buffers
        self.ro = ro
        self.b = b
        self.sds = [0 for _ in range(len(self.client_gradients))]
        self.zeno_amount_clients = int(
            self.cfg.federated_params.client_subset_size * (1 - self.b)
        )

    def find_sds(self):
        round_sds = [self.sds[rank] for rank in self.list_clients]
        self.initial_global_model_state = copy.deepcopy(self.global_model).state_dict()
        prev_test_loader = copy.deepcopy(self.test_loader)
        self.test_loader = [
            next(
                islice(
                    self.trust_loader,
                    self.cur_round % len(self.trust_loader),
                    (self.cur_round % len(self.trust_loader)) + 1,
                )
            )
        ]

        _, _, global_model_loss = self.model_trainer.server_eval_fn(self)

        for i, rank in enumerate(self.list_clients):
            client_loss = self.get_loss_for_sds(self.client_gradients[rank])
            grad_norm = self.find_grad_norm(self.client_gradients[rank])
            round_sds[i] = global_model_loss - client_loss - self.ro * (grad_norm**2)
            print(f"Client {rank} sds score: {round_sds[i]}")

        self.global_model.load_state_dict(self.initial_global_model_state)
        self.test_loader = prev_test_loader
        return round_sds

    def overwrite_server_global_model(self, grad_state_dict):
        tmp_weights = {}
        for key, weights in grad_state_dict.items():
            tmp_weights[key] = self.initial_global_model_state[key] + weights.to(
                self.device
            )
        self.global_model.load_state_dict(tmp_weights)

    def get_loss_for_sds(self, grad_state_dict):
        self.overwrite_server_global_model(grad_state_dict)
        _, _, client_loss = self.model_trainer.server_eval_fn(self)
        return client_loss

    def find_grad_norm(self, grad):
        needed_state = (
            self.global_model.state_dict().items()
            if self.use_buffers
            else self.global_model.named_parameters()
        )
        grad_1d = torch.cat([grad[key].flatten() for key, _ in needed_state])
        return torch.norm(grad_1d)

    def find_highest_sds(self):
        round_sds = self.find_sds()
        threshold = sorted(round_sds, reverse=True)[self.zeno_amount_clients - 1]
        round_sds = [value if value >= threshold else 0 for value in round_sds]
        print(
            f"Chosen clients: {[rank for i, rank in enumerate(self.list_clients) if round_sds[i]]}"
        )
        return round_sds
