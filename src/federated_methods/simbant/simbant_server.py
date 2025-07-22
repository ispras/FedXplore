import copy
import torch
import torch.nn.functional as F

from utils.data_utils import (
    get_dataset_loader,
    train_val_split,
)
from ..fltrust.fltrust_server import FLTrustServer


class SimBANTServer(FLTrustServer):
    def __init__(self, cfg, trust_df, trust_test_samples, prob_temperature):
        self.trust_test_df = copy.deepcopy(trust_df)
        self.trust_test_samples = trust_test_samples
        self.prob_temperature = prob_temperature

        train_val_prop = self.trust_test_samples / len(trust_df.data)
        trust_df.data, self.trust_test_df.data = train_val_split(
            trust_df.data,
            train_val_prop,
            random_state=cfg.random_state,
        )
        super().__init__(cfg, trust_df)
        self.prev_test_loader = self.test_loader

    def trust_eval_models(self):
        def set_gradients(gradients):
            new_weights = {}
            for key, weights in gradients.items():
                new_weights[key] = self.initial_global_model_state[key] + weights.to(
                    self.device
                )
            self.global_model.load_state_dict(new_weights)

        # Train server model
        self.fltrust_train()
        # Set test loader as a trust test part
        self.test_loader = get_dataset_loader(
            self.trust_test_df, self.cfg, drop_last=False
        )
        # Set server trust train updates
        set_gradients(self.server_grad)
        # Get server result
        targets, server_result, _ = self.model_trainer.server_eval_fn(self)
        # Get probabilities
        server_probs = F.softmax(
            torch.as_tensor(server_result) / self.prob_temperature, dim=-1
        )
        # Convert targets to tensor first
        targets_tensor = torch.as_tensor(targets)

        # Determine if we need one-hot encoding
        if targets_tensor.ndim == 1:
            # Single-dimension: class indices, need OHE
            ohe_targets = F.one_hot(
                targets_tensor.to(torch.long),
                num_classes=self.trust_df.num_classes
            ).float()
        elif targets_tensor.ndim == 2 and targets_tensor.shape[1] == self.trust_df.num_classes:
            # Already in OHE format: [batch_size, num_classes]
            ohe_targets = targets_tensor.float()
        else:
            raise ValueError(
                f"Unsupported target shape: {targets_tensor.shape}. "
                f"Expected 1D tensor or 2D tensor with {self.trust_df.num_classes} columns"
            )

        ohe_targets = ohe_targets.to(server_probs.device)
        # Get clients result
        clients_probs = []
        for rank in self.list_clients:
            client_grad = self.client_gradients[rank]
            set_gradients(client_grad)
            _, client_result, _ = self.model_trainer.server_eval_fn(self)

            client_probs = F.softmax(
                torch.as_tensor(client_result) / self.prob_temperature, dim=-1
            )
            clients_probs.append(client_probs)

        # Get back to initial state
        self.global_model.load_state_dict(self.initial_global_model_state)
        self.test_loader = self.prev_test_loader
        return ohe_targets, server_probs, clients_probs
