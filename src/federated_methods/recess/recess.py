from collections import OrderedDict
import time
from hydra.utils import instantiate

from ..byzantine_base.byzantine import ByzantineBase
from .recess_server import RecessServer
from utils.attack_utils import set_client_map_round


class Recess(ByzantineBase):
    def __init__(
        self,
        baseline_decreased_score: int,
        init_trust_score: int,
    ):
        super().__init__()
        self.baseline_decreased_score = baseline_decreased_score
        self.init_trust_score = init_trust_score

    def count_trust_scores(self):
        self.server.calculate_trust_scores()
        return self.server.trust_scores

    def _init_server(self, cfg):
        assert "SGD" in str(
            cfg.optimizer._target_
        ), f"Recess works only with SGD client optimizer. You provide: {cfg.optimizer._target_}"
        self.server = RecessServer(
            cfg, self.baseline_decreased_score, self.init_trust_score
        )

    # def _modify_gradients(self):
    #     # Modify each client's gradient based on there trust score
    #     for i, rank in enumerate(self.list_clients):
    #         print(f"Client {rank} trust score: {self.server.trust_scores[i]}")
    #         modified_client_model_weights = OrderedDict()
    #         for key, weights in self.server.client_gradients[rank].items():
    #             modified_client_model_weights[key] = (
    #                 self.server.trust_scores[i]
    #                 * weights
    #                 * self.cfg.federated_params.amount_of_clients
    #             )
    #         self.server.client_gradients[i] = modified_client_model_weights

    def get_communication_content(self, rank):
        client_state = self.server.adjust_model(rank)
        return {
            "update_model": {k: v.cpu() for k, v in client_state.items()},
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
        }

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches
        self.server.global_model = instantiate(
            self.cfg.model, num_classes=self.df.num_classes
        )

        for cur_round in range(self.rounds):
            print(f"\nRound number: {cur_round}")
            begin_round_time = time.time()
            self.cur_round = cur_round
            self.server.cur_round = cur_round

            self.server.test_global_model()

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_attack_map,
                self.attack_rounds,
                self.attack_scheme,
                cur_round,
            )
            self.list_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            self.list_clients.sort()
            self.server.list_clients = self.list_clients
            print(f"Clients on this communication: {self.list_clients}")
            print(
                f"Amount of clients on this communication: {len(self.list_clients)}\n"
            )

            # ======================RECESS=======================
            self.server.gradient_resetting()
            # ======================RECESS=======================

            self.train_round()

            # We save model after first RECESS communication to compare with original server state
            self.server.save_best_model(cur_round)

            # ======================RECESS=======================
            self.server.gradient_normalization()
            # ======================RECESS=======================

            self.train_round()

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        print("Shutdown clients, federated learning end", flush=True)
        self.manager.stop_train()
