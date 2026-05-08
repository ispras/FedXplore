from types import MethodType

import numpy as np
import pandas as pd
from hydra.utils import instantiate

from utils.data_utils import get_dataset_loader


class AttackClient:
    def apply_attack(self, client_instance):
        """Apply attack functionality to existing client instance

        Args:
            client_instance (Client): Client instance that is set for attack

        Returns:
            client_instance (Client): Client instance with added attack functionality
        """
        raise NotImplementedError


class LabelFlipClient(AttackClient):
    def __init__(self, percent_of_changed_labels):
        self.percent_of_changed_labels = percent_of_changed_labels

    def apply_attack(self, client_instance):
        """LabelFlip attack corrupts some percent of client labels. It apply attack functionality in 2 steps:
        1. Change `train_df` client attribute to the same dataframe with corrupted labels
        2. Reinit `train_loader` client attribute with corrupted `train_df` to set up training process with attack.

        Args:
            client_instance (Client): Client instance that is set for attack

        Returns:
            client_instance (Client): Client instance with added LabelFlip attack functionality
        """
        client_instance.train_dataset.data = self._change_client_labels(
            client_instance.train_dataset.data,
            client_instance.rank,
        )

        client_instance.valid_dataset.data = self._change_client_labels(
            client_instance.valid_dataset.data,
            client_instance.rank,
        )

        """
        TODO issue205: Sync ECG dataset targets after label corruption.
        If the task uses an ECG dataset, update the internal
        ecg_dataset.target array to match the modified
        train_dataset.data.target and valid_dataset.data.target. The
        project stores targets both in the dataframe and inside the
        ecg_dataset object; the ECG trainer reads targets from
        ecg_dataset, so failing to sync them causes training to use
        stale labels.
        """
        if hasattr(client_instance.train_dataset, "ecg_dataset"):
            client_instance.train_dataset.ecg_dataset.target = (
                client_instance.train_dataset.data.target.values
            )

        if hasattr(client_instance.valid_dataset, "ecg_dataset"):
            client_instance.valid_dataset.ecg_dataset.target = (
                client_instance.valid_dataset.data.target.values
            )

        client_instance.train_loader = get_dataset_loader(
            client_instance.train_dataset, client_instance.cfg, drop_last=False
        )

        client_instance.valid_loader = get_dataset_loader(
            client_instance.valid_dataset, client_instance.cfg, drop_last=False
        )
        return client_instance

    def _change_client_labels(self, train_df, rank):
        rng = np.random.RandomState(int(rank))
        labels = np.asarray(train_df["target"].tolist(), dtype=np.int64)

        attacked_labels = rng.choice(
            np.prod(labels.shape),
            int(self.percent_of_changed_labels * np.prod(labels.shape)),
            replace=False,
        )

        if labels.ndim > 1:
            # multilabel case
            labels.flat[attacked_labels] -= 1
            labels = np.abs(labels)
            train_df["target"] = pd.Series(
                [row.tolist() for row in labels], index=train_df.index, dtype=object
            )
        else:
            # otherwise we assume multiclass
            num_classes = np.unique(labels).size
            corrupted_labels = rng.randint(0, num_classes, size=attacked_labels.size)
            labels.flat[attacked_labels] = corrupted_labels.flat
            train_df["target"] = pd.Series(labels.tolist(), index=train_df.index)

        return train_df


class AttackGradClient(AttackClient):
    def __init__(self, percent_of_changed_grads):
        self.percent_of_changed_grads = percent_of_changed_grads

    def apply_attack(self, client_instance):
        """Gradient attack corrupts some percent of client gradients. It apply attack functionality in 2 steps:
        1. Set `percent_of_changed_grads` as client instance attribute to use it in get_grad client method.
        2. Change `get_grad` method to attack version.

        Args:
            client_instance (Client): Client instance that is set for attack

        Returns:
            client_instance (Client): Client instance with added gradient attack functionality
        """
        client_instance.percent_of_changed_grads = self.percent_of_changed_grads
        client_instance.get_grad = MethodType(
            AttackGradClient.get_grad, client_instance
        )
        client_instance.grad_attack = MethodType(self.grad_attack(), client_instance)
        client_instance.load_corrupted_model = MethodType(
            self.load_corrupted_model(), client_instance
        )
        return client_instance

    def get_grad(self):
        print(f"Сейчас атакует клиент {self.rank}", flush=True)
        rng = np.random.RandomState(self.rank)
        self.model.eval()
        random_model = instantiate(
            self.cfg.model, num_classes=self.train_dataset.num_classes
        )
        self.random_weights = random_model.state_dict()
        for name, param in self.model.named_parameters():
            param_final_grad = param.data - self.server_model_state[name]
            param_final_grad_flat = param_final_grad.view(-1)
            self.flip_ids = rng.choice(
                param_final_grad_flat.numel(),
                int(param_final_grad_flat.numel() * self.percent_of_changed_grads),
                replace=False,
            )
            self.param = param
            self.name = name
            param_final_grad_flat[self.flip_ids] = self.grad_attack()
            self.grad[name] = param_final_grad_flat.view_as(param_final_grad).to("cpu")
        for name, buffer in self.model.named_buffers():
            self.grad[name] = buffer.to("cpu") - self.server_model_state[name].to("cpu")

        # For PoW CS we need to eval model after training,
        # so we load corrupted model for this
        if "pow" in self.cfg.client_selector._target_:
            self.load_corrupted_model()

    def grad_attack(self):
        raise NotImplementedError("Subclasses must implement this method")

    def load_corrupted_model(self):
        raise NotImplementedError("Subclasses must implement this method")


class SignFlipClient(AttackGradClient):
    def grad_attack(self):
        def grad_flip(self):
            # Gradient Attacking with sign flipping
            return (
                self.server_model_state[self.name].view(-1)[self.flip_ids]
                - self.param.data.view(-1)[self.flip_ids]
            )

        # This method returns a closure that dynamically adjusts the gradients for custom method
        return grad_flip

    def load_corrupted_model(self):
        def load_sign_flip_model(self):
            self.model.eval()
            loading_weights = self.model.state_dict()
            for k, v in loading_weights.items():
                loading_weights[k] = self.server_model_state[k].to("cpu") - self.grad[k]

            self.model.load_state_dict(loading_weights)

        return load_sign_flip_model


class RandomGradClient(AttackGradClient):
    def grad_attack(self):
        def grad_random(self):
            # Gradient Attacking with randomization
            return (
                self.random_weights[self.name].view(-1)[self.flip_ids].to(self.device)
                - self.server_model_state[self.name].view(-1)[self.flip_ids]
            )

        # This method returns a closure that dynamically adjusts the gradients for custom method
        return grad_random

    def load_corrupted_model(self):
        def load_random_grad_model(self):
            self.model.eval()
            loading_weights = self.model.state_dict()
            for k, v in loading_weights.items():
                loading_weights[k] = self.server_model_state[k].to("cpu") + self.grad[k]

            self.model.load_state_dict(loading_weights)

        return load_random_grad_model


class IPM(AttackClient):
    """IPM (https://dprg.cs.uiuc.edu/data/files/2019/uai2019_byz.pdf)
    We apply IPM attack directly on server because our core architecture (see C4.md docs)
    does not support communications between clients.
    """

    def __init__(self, ipm_eps):
        self.ipm_eps = ipm_eps

    def apply_attack(self, client_instance):
        return client_instance

    def load_corrupted_model(self):
        def load_ipm_model(self):
            pass

        return load_ipm_model


class ALIE(AttackClient):
    """ALIE (https://arxiv.org/pdf/1902.06156)
    We apply ALIE attack directly on server because our core architecture (see C4.md docs)
    does not support communications between clients.
    AttackClient class support labelflip backdooring.
    """

    def __init__(
        self, percent_of_changed_grads, attack_type, percent_of_changed_labels
    ):
        assert attack_type in [
            "random_grad",
            "label_flip",
        ], f"at the current moment only ['random_grad', 'label_flip'] backdooring is supported, you provide: {attack_type}"
        self.attack_type = attack_type
        self.percent_of_changed_grads = percent_of_changed_grads
        self.percent_of_changed_labels = percent_of_changed_labels

    def apply_attack(self, client_instance):
        """ALIE (https://arxiv.org/pdf/1902.06156) attack imitates the majority attacking clients, making it difficult to defend against.
        It apply attack functionality in 3 steps:
        1. define get_grad
        2. init attack attributes
        3. corrupt train_loader if use label_flip backdooring

        Args:
            client_instance (Client): Client instance that is set for attack

        Returns:
            client_instance (Client): Client instance with added mitm attack functionality
        """
        # Define get_grad
        client_instance.get_grad = MethodType(ALIE.get_grad, client_instance)
        client_instance.get_random_grad = MethodType(
            ALIE.get_random_grad, client_instance
        )
        client_instance.get_true_grad = MethodType(ALIE.get_true_grad, client_instance)

        # init attributes
        client_instance.attack_type = self.attack_type
        client_instance.percent_of_changed_grads = self.percent_of_changed_grads

        # define corrupted train_loader
        if self.attack_type == "label_flip":
            client_instance.train_df.data = self.change_client_labels(
                client_instance.train_df.data,
                client_instance.train_df.name,
                client_instance.rank,
            )
            client_instance.train_loader = get_dataset_loader(
                client_instance.train_df, client_instance.cfg, drop_last=False
            )

        client_instance.load_corrupted_model = MethodType(
            self.load_corrupted_model(), client_instance
        )
        return client_instance

    def get_random_grad(self):
        rng = np.random.RandomState(self.rank)
        self.model.eval()
        random_model = instantiate(
            self.cfg.model, num_classes=self.train_dataset.num_classes
        )
        self.random_weights = random_model.state_dict()
        for name, param in self.model.named_parameters():
            param_final_grad = param.data - self.server_model_state[name]
            param_final_grad_flat = param_final_grad.view(-1)
            self.flip_ids = rng.choice(
                param_final_grad_flat.numel(),
                int(param_final_grad_flat.numel() * self.percent_of_changed_grads),
                replace=False,
            )
            self.param = param
            self.name = name
            if self.attack_type == "random_grad":
                param_final_grad_flat[self.flip_ids] = (
                    self.random_weights[self.name]
                    .view(-1)[self.flip_ids]
                    .to(self.device)
                    - self.server_model_state[self.name].view(-1)[self.flip_ids]
                )
            self.grad[name] = param_final_grad_flat.view_as(param_final_grad).to("cpu")
        for name, buffer in self.model.named_buffers():
            self.grad[name] = buffer.to("cpu") - self.server_model_state[name].to("cpu")

    def get_true_grad(self):
        self.model.eval()
        for key, _ in self.model.state_dict().items():
            self.grad[key] = self.model.state_dict()[key].to(
                "cpu"
            ) - self.server_model_state[key].to("cpu")

    def get_grad(self):
        if self.attack_type == "random_grad":
            self.get_random_grad()
        else:
            self.get_true_grad()

        # For PoW CS we need to eval model after training,
        # so we load corrupted model for this
        if "pow" in self.cfg.client_selector._target_:
            self.load_corrupted_model()

    def change_client_labels(self, train_df, data_name, rank):
        # Seed randomization in accordance with client rank. See https://github.com/numpy/numpy/issues/9248
        rng = np.random.RandomState(rank)
        labels = np.array(train_df["target"].tolist())
        num_classes = train_df["target"].nunique()
        attacked_labels = rng.choice(
            np.prod(labels.shape),
            int(self.percent_of_changed_labels * np.prod(labels.shape)),
            replace=False,
        )
        corrupted_labels = rng.randint(0, num_classes, size=attacked_labels.size)
        labels.flat[attacked_labels] = corrupted_labels
        train_df.loc[train_df.index, "target"] = np.abs(labels)
        if not "cifar" in data_name:
            train_df.loc[train_df.index, "target"] = train_df["target"].apply(
                lambda x: [x]
            )
        return train_df

    def load_corrupted_model(self):
        def load_random_grad_model(self):
            self.model.eval()
            loading_weights = self.model.state_dict()
            for k, v in loading_weights.items():
                loading_weights[k] = self.server_model_state[k].to("cpu") + self.grad[k]

            self.model.load_state_dict(loading_weights)

        return load_random_grad_model
