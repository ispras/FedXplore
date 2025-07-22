import copy
import numpy as np
from types import MethodType
from .base import BaseSelector
from collections import OrderedDict
from utils.model_utils import summ_dicts
from utils.model_utils import diff_dicts
from utils.model_utils import square_diff_dicts
from utils.model_utils import net_dict_weights_norm


class Delta(BaseSelector):
    def __init__(self, cfg, alpha_1, alpha_2):
        super().__init__(cfg)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        if num_clients_subset == self.amount_of_clients:
            return list(range(self.amount_of_clients))

        clients = list(range(self.amount_of_clients))
        selected_clients = np.random.choice(
            clients, size=num_clients_subset, replace=False, p=self.client_probs
        ).tolist()

        return selected_clients

    def change_functionality(self, trainer):
        # Setup server side parametrs
        trainer.server.alpha_1 = self.alpha_1
        trainer.server.alpha_2 = self.alpha_2

        trainer.server.client_probs = np.array(
            [
                1.0 / trainer.server.amount_of_clients
                for _ in range(trainer.server.amount_of_clients)
            ]
        )
        print(f"Initial clients probabilities:\n{trainer.server.client_probs}")

        trainer.server.client_sigmas = [
            None for _ in range(trainer.server.amount_of_clients)
        ]

        # Update aggregate function
        trainer.orig_aggregate = MethodType(
            getattr(type(trainer), "aggregate"),
            trainer,
        )
        trainer.aggregate = MethodType(
            Delta.aggregate,
            trainer,
        )

        # Setup update_probs function
        trainer.server.update_probs = MethodType(Delta.update_probs, trainer.server)

        # Change set_client_result function
        trainer.server.orig_set_client_result = MethodType(
            getattr(type(trainer.server), "set_client_result"),
            trainer.server,
        )
        trainer.server.set_client_result = MethodType(
            Delta.set_client_result,
            trainer.server,
        )

        # Add initialization parametr to client init function
        trainer.client_cls.orig_init_ = trainer.client_cls.__init__
        trainer.client_cls.__init__ = Delta.client__init__

        # Change get_communication_content in client side
        trainer.client_cls.orig_get_communication_content = (
            trainer.client_cls.get_communication_content
        )
        trainer.client_cls.get_communication_content = Delta.get_communication_content

        # Add functions to client
        trainer.client_cls.get_grad_by_batch = Delta.get_grad_by_batch
        trainer.client_cls.get_sigma = Delta.get_sigma

        # Change train on client side
        trainer.client_cls.orig_train = trainer.client_cls.train
        trainer.client_cls.train = Delta.train

        # Setup select_clients function
        trainer.server.select_clients_to_train = MethodType(
            Delta.select_clients_to_train, trainer.server
        )

        return trainer

    def aggregate(self):
        self.server.client_probs = self.server.update_probs(self.list_clients)
        print(f"\nClients probabilities:")
        for i, prob in enumerate(self.server.client_probs):
            print(f"{i} : {prob}")

        return self.orig_aggregate()

    def update_probs(self, participated_clients):
        self.no_buffer_state_dict = [
            name
            for name, param in self.global_model.named_parameters()
            if param.requires_grad
        ]

        new_probs = copy.deepcopy(self.client_probs)

        N = len(participated_clients)
        # 1 / n sum_i g_it
        nabla_hat_f = {param_name: None for param_name in self.no_buffer_state_dict}

        # all g_it
        hat_g = [None for i in range(self.amount_of_clients)]

        for rank in participated_clients:
            hat_g[rank] = self.client_gradients[rank]
            nabla_hat_f = summ_dicts(
                nabla_hat_f, hat_g[rank], self.no_buffer_state_dict
            )

        # get 1/n for nabla hat_f
        for param_name in self.no_buffer_state_dict:
            nabla_hat_f[param_name] = 1 / N * nabla_hat_f[param_name]

        sqrts = [None for _ in range(self.amount_of_clients)]
        for rank in participated_clients:
            dzeta_i = net_dict_weights_norm(
                diff_dicts(hat_g[rank], nabla_hat_f, self.no_buffer_state_dict)
            )
            sqrts[rank] = np.sqrt(
                self.alpha_1 * dzeta_i**2 + self.alpha_2 * self.client_sigmas[rank] ** 2
            )

        summed_probs = sum(
            [
                self.client_probs[rank]
                for rank, _ in enumerate(sqrts)
                if sqrts[rank] is None
            ]
        )
        for rank in participated_clients:
            new_probs[rank] = (
                sqrts[rank]
                / sum([x for x in sqrts if x is not None])
                * (1 - summed_probs)
            )

        return new_probs

    def set_client_result(self, client_result):
        self.orig_set_client_result(client_result)
        self.client_sigmas[client_result["rank"]] = client_result["sigma"]

    def client__init__(self, *client_args, **client_kwargs):
        self.orig_init_(*client_args, **client_kwargs)
        self.batch_grads = []

    def get_communication_content(self):
        content = self.orig_get_communication_content()
        content["batch_grads"] = None
        content["sigma"] = self.sigma
        return content

    def get_grad_by_batch(self):
        self.model.train()
        for batch in self.train_loader:
            _, (input, targets) = batch

            inp = input[0].to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inp)

            loss = self.get_loss_value(outputs, targets)

            loss.backward()

            inp = input[0].to("cpu")
            targets = targets.to("cpu")

            # Collecting gradients
            self.model.eval()
            batch_grad = OrderedDict()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    batch_grad[name] = param.grad.to("cpu")

            self.batch_grads.append(batch_grad)
            self.model.train()

    def get_sigma(self):
        self.no_buffer_state_dict = [
            name for name, param in self.model.named_parameters() if param.requires_grad
        ]

        summed_g_hat_b_i = {
            param_name: None for param_name in self.no_buffer_state_dict
        }

        for i in range(len(self.batch_grads)):
            summed_g_hat_b_i = summ_dicts(
                summed_g_hat_b_i, self.batch_grads[i], self.no_buffer_state_dict
            )

        B = len(self.batch_grads)
        for param_name in self.no_buffer_state_dict:
            summed_g_hat_b_i[param_name] = 1 / B * summed_g_hat_b_i[param_name]

        sigma_i = {param_name: None for param_name in self.no_buffer_state_dict}
        for i in range(len(self.batch_grads)):
            sigma_i = summ_dicts(
                sigma_i,
                square_diff_dicts(
                    self.batch_grads[i], summed_g_hat_b_i, self.no_buffer_state_dict
                ),
                self.no_buffer_state_dict,
            )

        for param_name in self.no_buffer_state_dict:
            sigma_i[param_name] = (1 / B * sigma_i[param_name]) ** 0.5

        sigma_i_norm = net_dict_weights_norm(sigma_i)

        return sigma_i_norm

    def train(self):
        self.orig_train()

        self.batch_grads = []
        self.get_grad_by_batch()
        self.sigma = self.get_sigma()
