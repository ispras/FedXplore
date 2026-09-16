from __future__ import annotations

import unittest
from pathlib import Path

from ui.create_run import initial_dataset_roles, validate_experiment_state
from ui.research_catalog import load_research_catalog, metadata_for, ordered_options


CATALOG_PATH = Path(__file__).resolve().parents[1] / "ui" / "research_catalog.yaml"


class ResearchCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_research_catalog(CATALOG_PATH)

    def test_alias_resolution_and_tags(self) -> None:
        method = metadata_for(self.catalog, "federated_method", "fed_avg")
        self.assertEqual(method["display_name"], "FedAvg")
        self.assertEqual(method["tags"], ["Heterogeneity"])

    def test_unknown_component_has_safe_fallback(self) -> None:
        unknown = metadata_for(self.catalog, "client_selector", "my_new_selector")
        self.assertEqual(unknown["display_name"], "My New Selector")
        self.assertFalse(unknown["known"])
        self.assertIn("No curated description", unknown["description"])

    def test_toy_components_have_curated_ui_context(self) -> None:
        dataset = metadata_for(self.catalog, "dataset", "synthetic_2d")
        attack = metadata_for(self.catalog, "attack", "binary_label_flip")
        model = metadata_for(self.catalog, "model", "factorized_linear")

        self.assertTrue(dataset["known"])
        self.assertIn("Gaussian", dataset["description"])
        self.assertTrue(attack["known"])
        self.assertIn("swap labels", attack["description"])
        self.assertTrue(model["known"])
        self.assertIn("representation", model["description"])

    def test_order_keeps_unknown_local_components(self) -> None:
        result = ordered_options(self.catalog, "attack", ["ipm", "custom_attack", "no_attack"])
        self.assertEqual(result, ["no_attack", "ipm", "custom_attack"])


class CreateRunStateTests(unittest.TestCase):
    def test_dataset_roles_are_initialized_once(self) -> None:
        self.assertEqual(
            initial_dataset_roles("cifar10", ""),
            {"train_dataset": "cifar10", "test_dataset": "cifar10", "trust_dataset": ""},
        )
        self.assertIsNone(initial_dataset_roles("cifar10", "cifar10"))

    def test_validation_reports_trust_warning_and_client_error(self) -> None:
        errors, warnings = validate_experiment_state(
            {
                "ui_base__federated_params_amount_of_clients": 10,
                "ui_base__federated_params_client_subset_size": 11,
                "ui_base__federated_params_communication_rounds": 1,
                "ui_base__federated_params_local_epochs": 1,
                "ui_trust_dataset": "",
            },
            requires_trust_dataset=True,
        )
        self.assertTrue(any("cannot exceed" in error for error in errors))
        self.assertTrue(any("trust dataset" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
