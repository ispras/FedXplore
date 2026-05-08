import numpy as np
import warnings

from ..fedavg.fedavg import FedAvg


class FedNova(FedAvg):
    def __init__(self):
        super().__init__()

    def _init_federated(self, cfg):
        super()._init_federated(cfg)
        if "SGD" not in str(cfg.optimizer._target_):
            warnings.warn(
                f"\nFedNova is designed for use with the SGD optimizer.\nBehavior with other optimizers is not defined and may be incorrect.\nYour optimizer: {cfg.optimizer._target_}",
                UserWarning,
            )

    def calculate_aggregation_weights(self):
        # We implement FedNova for SGD client optimizer
        # The article did not describe how to apply the method with Adam,
        # but it can be done, we will implement it in the future

        cur_clients_df_len = self.clients_df_len[self.list_clients]

        # Calculate tau_i (amount client local iteration of gradient descent)
        # We assume that it is constant every round
        clients_local_iterations = np.ceil(
            cur_clients_df_len / self.cfg.training_params.batch_size
        )

        # Calculate p_i (client weights in aggregation)
        fedavg_weights = cur_clients_df_len / sum(cur_clients_df_len)
        final_weights = (
            fedavg_weights
            * sum(fedavg_weights * clients_local_iterations)
            / clients_local_iterations
        ).tolist()

        return final_weights
