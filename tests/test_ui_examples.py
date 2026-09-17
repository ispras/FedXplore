from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from ui.examples import (
    THREAD_LIMITS,
    available_preferred_metrics,
    build_example_launch_plan,
    build_group_id,
    example_context_from_specs,
    launch_example_suite,
    load_examples,
)
from ui.run_ui import comparison_run_label, saved_override_values


EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "ui" / "examples.yaml"


class ExampleCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = load_examples(EXAMPLES_PATH)

    def test_catalog_contains_exactly_the_two_curated_suites(self) -> None:
        self.assertEqual(set(self.examples), {"interdependency", "personalization"})

    def test_every_example_has_an_existing_poster(self) -> None:
        for example in self.examples.values():
            poster_path = EXAMPLES_PATH.parent / example.data["poster"]
            self.assertTrue(poster_path.is_file(), poster_path)

    def test_interdependency_plan_matches_the_canonical_script(self) -> None:
        example = self.examples["interdependency"]
        plan = build_example_launch_plan(
            example,
            group_id="interdependency_toy_20260916_120000",
            device="cuda",
            gpu_ids=[0],
            seed=42,
            tracking_uri="http://tracking:5000",
            base_env={"EXISTING": "1"},
        )
        self.assertEqual(len(plan), 5)
        self.assertEqual(
            [request.condition for request in plan],
            ["clean_cc_uniform", "labelflip_cc_uniform", "labelflip_cc_pow", "labelflip_cc_fedcbs", "labelflip_fedavg_uniform"],
        )
        first = plan[0]
        for override in [
            "dataset@train_dataset=synthetic_2d",
            "model=logistic_regression",
            "optimizer.lr=0.1",
            "distribution.alpha=2.0",
            "federated_params.amount_of_clients=20",
            "federated_params.client_subset_size=10",
            "federated_params.communication_rounds=40",
            "manager.batch_generator.batch_size=10",
            "federated_method.tau_clip=0.2",
            "random_state=42",
            "training_params.device=cuda",
            "training_params.device_ids=[0]",
        ]:
            self.assertIn(override, first.overrides)
        self.assertIn("federated_params.prop_attack_clients=0.35", plan[1].overrides)
        self.assertIn("client_selector.candidate_set_size=15", plan[2].overrides)
        self.assertEqual(first.subprocess_env["EXISTING"], "1")
        self.assertTrue(all(first.subprocess_env[key] == value for key, value in THREAD_LIMITS.items()))

    def test_personalization_plan_matches_the_canonical_script(self) -> None:
        example = self.examples["personalization"]
        plan = build_example_launch_plan(
            example,
            group_id="personalization_toy_20260916_120000",
            device="cpu",
            gpu_ids=[9],
            seed=42,
            tracking_uri="/tmp/mlruns",
            base_env={},
        )
        self.assertEqual(
            [request.condition for request in plan],
            ["fedavg", "personalized_local", "ditto", "pfedme", "fedrep", "fedamp"],
        )
        first = plan[0]
        for override in [
            "dataset@train_dataset=personalization_2d",
            "model=factorized_linear",
            "optimizer.lr=0.08",
            "distribution=uniform",
            "federated_params.amount_of_clients=10",
            "federated_params.client_subset_size=10",
            "federated_params.communication_rounds=20",
            "federated_params.local_epochs=1",
            "federated_params.client_train_val_prop=0.25",
            "manager.batch_generator.batch_size=5",
            "training_params.device=cpu",
            "training_params.device_ids=[]",
        ]:
            self.assertIn(override, first.overrides)
        self.assertIn("federated_method.proximity=0.0", plan[1].overrides)

    def test_every_request_has_mlflow_and_durable_batch_metadata(self) -> None:
        example = self.examples["interdependency"]
        plan = build_example_launch_plan(
            example,
            group_id=build_group_id(example, datetime(2026, 9, 16, 12, 0, 0)),
            device="cpu",
            gpu_ids=[],
            seed=7,
            tracking_uri="/tmp/mlruns",
            base_env={},
        )
        for index, request in enumerate(plan, start=1):
            self.assertIn("logger=mlflow", request.overrides)
            self.assertIn("logger.experiment_name=FedXplore Toy Examples", request.overrides)
            self.assertIn("logger.tracking_uri=/tmp/mlruns", request.overrides)
            self.assertIn(f"+logger.tags.seed=7", request.overrides)
            batch = request.spec_data["example_batch"]
            self.assertEqual(batch["run_index"], index)
            self.assertEqual(batch["run_count"], 5)
            self.assertEqual(batch["group_id"], "interdependency_toy_20260916_120000")

    def test_context_and_preferred_metrics_are_safe_for_mixed_runs(self) -> None:
        plan = build_example_launch_plan(
            self.examples["personalization"],
            group_id="personalization_toy_20260916_120000",
            device="cpu", gpu_ids=[], seed=42, tracking_uri="/tmp/mlruns", base_env={},
        )
        context = example_context_from_specs([request.spec_data for request in plan])
        self.assertIsNotNone(context)
        self.assertEqual(
            available_preferred_metrics(context or {}, ["test/Accuracy_overall", "val/Accuracy_overall"]),
            ["val/Accuracy_overall", "test/Accuracy_overall"],
        )
        self.assertIsNone(example_context_from_specs([plan[0].spec_data, {}]))

    def test_partial_batch_failures_keep_successful_normal_runs(self) -> None:
        plan = build_example_launch_plan(
            self.examples["interdependency"],
            group_id="interdependency_toy_20260916_120000",
            device="cpu", gpu_ids=[], seed=42, tracking_uri="/tmp/mlruns", base_env={},
        )
        calls: list[str] = []

        def launch(_root, run_name, *_args, **_kwargs):
            calls.append(run_name)
            if len(calls) == 4:
                raise RuntimeError("synthetic start failure")
            return {"run_id": f"run-{len(calls)}", "run_name": run_name}

        statuses, errors = launch_example_suite(
            plan, launch, repo_root=Path("/tmp/repo"), mlflow_url="http://localhost:5000"
        )
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(statuses), 4)
        self.assertEqual(len(errors), 1)

    def test_override_metadata_and_compare_label_have_example_fallbacks(self) -> None:
        values = saved_override_values(
            {"overrides": ["logger=mlflow", "dataset@train_dataset=synthetic_2d", "federated_params.communication_rounds=40"]}
        )
        self.assertEqual(values["logger"], "mlflow")
        self.assertEqual(values["dataset@train_dataset"], "synthetic_2d")
        self.assertEqual(
            comparison_run_label(
                {"run_id": "ordinary-run-id"},
                {"name": "ignored", "spec": {"example_batch": {"run_label": "FedAvg"}}},
            ),
            "FedAvg",
        )


if __name__ == "__main__":
    unittest.main()
