from __future__ import annotations

import unittest

from ui.create_run import build_experiment_summary, readable_label, readable_option


class CreateRunPresentationTests(unittest.TestCase):
    def test_readable_label_uses_special_cases_and_humanizes_paths(self) -> None:
        self.assertEqual(readable_label("optimizer.weight_decay"), "Weight decay")
        self.assertEqual(readable_label("config.ignore_index"), "Ignore index")
        self.assertEqual(readable_label("federated_params.print_client_metrics"), "Print client metrics")
        self.assertEqual(readable_label("custom_long_option"), "Custom long option")

    def test_summary_contains_only_research_level_values(self) -> None:
        summary = dict(
            build_experiment_summary(
                {
                    "ui_train_dataset": "cifar10",
                    "ui_model": "resnet18",
                    "ui_distribution": "dirichlet",
                    "ui_comp__distribution__dirichlet__alpha": 0.1,
                    "ui_federated_method": "fedavg",
                    "ui_client_selector": "fedcbs",
                    "ui_optimizer": "adam",
                    "ui_comp__optimizer__adam__lr": 0.003,
                    "ui_attack_type": "no_attack",
                    "ui_base__random_state": 42,
                    "ui_base__federated_params_amount_of_clients": 100,
                    "ui_base__federated_params_client_subset_size": 25,
                }
            )
        )
        self.assertEqual(summary["Dataset"], "Cifar10")
        self.assertEqual(summary["Federation"], "100 clients · 25/round")
        self.assertEqual(summary["Distribution"], "Dirichlet α = 0.1")
        self.assertEqual(summary["Optimizer"], "Adam · lr=0.003")
        self.assertEqual(summary["Attack"], "No Attack")

    def test_readable_option_handles_blank_values(self) -> None:
        self.assertEqual(readable_option("central_clip"), "Central Clip")
        self.assertEqual(readable_option(""), "Not configured")


if __name__ == "__main__":
    unittest.main()
