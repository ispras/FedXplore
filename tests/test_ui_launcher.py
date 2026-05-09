from __future__ import annotations

import shlex
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from ui.launcher import (
    build_command,
    build_overrides,
    build_subprocess_env,
    create_run_dir,
    extract_mlflow_run_url_from_text,
    format_shell_command,
    is_pid_alive,
    load_templates,
    parse_iso_datetime,
    parse_raw_overrides,
    read_status,
    write_status,
)


class LauncherTests(unittest.TestCase):
    def make_repo_root(self, tmp_path: Path) -> Path:
        repo_root = tmp_path / "repo"
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/train.py").write_text("print('ok')\n", encoding="utf-8")
        return repo_root

    def test_build_command_generates_expected_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = self.make_repo_root(Path(tmp_dir))
            overrides = ["federated_method=fedavg", "training_params.batch_size=32"]

            cmd = build_command(repo_root, overrides)

            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[1], "src/train.py")
            self.assertEqual(cmd[2:], overrides)

    def test_build_overrides_maps_form_values(self) -> None:
        form_values = {
            "run_name": "demo",
            "random_state": 41,
            "communication_rounds": 5,
            "dataset": "cifar10",
            "trust_dataset": "",
            "distribution": "dirichlet",
            "distribution_alpha": 0.1,
            "amount_of_clients": 10,
            "client_subset_size": 5,
            "training_batch_size": 64,
            "manager_batch_size": 5,
            "device_mode": "gpu",
            "device_ids": "0,1",
            "print_client_metrics": False,
            "federated_method": "fedavg",
            "client_selector": "uniform",
            "logger": "mlflow",
            "mlflow_tracking_uri": "http://10.100.202.109:5000/",
            "mlflow_experiment_name": "exp_name",
        }

        overrides = build_overrides(form_values, ["model=resnet18"])

        self.assertIn("distribution.alpha=0.1", overrides)
        self.assertIn("training_params.device_ids=[0,1]", overrides)
        self.assertIn("logger.run_name=demo", overrides)
        self.assertEqual(overrides[-1], "model=resnet18")

    def test_shell_command_preview_roundtrips(self) -> None:
        cmd = ["python3", "src/train.py", "logger.experiment_name=run name"]
        preview = format_shell_command(cmd)
        self.assertEqual(shlex.split(preview), cmd)

    def test_build_subprocess_env_adds_no_proxy_hosts(self) -> None:
        env, hosts = build_subprocess_env(
            disable_proxy=True,
            mlflow_tracking_uri="http://10.100.202.109:5000/",
        )

        self.assertIn("10.100.202.109", hosts)
        self.assertIn("10.100.151.14", hosts)
        self.assertIn("10.100.202.109", env["NO_PROXY"])
        self.assertNotIn("HTTP_PROXY", env)

    def test_extract_mlflow_run_url_from_text(self) -> None:
        text = (
            "some text\n"
            "MLFLOW_RUN_ID=abc\n"
            "MLFLOW_RUN_URL=http://10.100.202.109:5000/#/experiments/168/runs/abc\n"
        )
        self.assertEqual(
            extract_mlflow_run_url_from_text(text),
            "http://10.100.202.109:5000/#/experiments/168/runs/abc",
        )

    def test_parse_iso_datetime_supports_timezone_iso(self) -> None:
        parsed = parse_iso_datetime("2026-05-09T15:08:32+03:00")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2026)
        self.assertEqual(parsed.minute, 8)

    def test_build_overrides_skips_empty_optional_groups(self) -> None:
        overrides = build_overrides(
            {
                "run_name": "demo",
                "selected_groups": {
                    "model": "resnet18",
                    "logger": "base",
                    "train_dataset": "cifar10",
                    "test_dataset": "cifar10",
                    "trust_dataset": "",
                    "distribution": "uniform",
                    "model_trainer": "image",
                    "federated_method": "fedavg",
                    "client_selector": "uniform",
                    "manager": "base_manager",
                    "loss": "ce",
                    "optimizer": "adam",
                    "preaggregator": "",
                    "manager_batch_generator": "sequential",
                },
                "base_params": {"random_state": 42},
                "component_params": {},
            },
            [],
        )

        self.assertNotIn("dataset@trust_dataset=null", overrides)
        self.assertNotIn("preaggregator=null", overrides)

    def test_is_pid_alive_rejects_invalid_pid(self) -> None:
        self.assertFalse(is_pid_alive(-1))

    @unittest.skipUnless(find_spec("yaml") is not None, "PyYAML is not installed")
    def test_load_templates_reads_form_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            template_dir = Path(tmp_dir)
            (template_dir / "debug.yaml").write_text(
                "name: Debug\n"
                "description: Sample\n"
                "form:\n"
                "  run_name: sample\n"
                "overrides:\n"
                "  - model=resnet18\n",
                encoding="utf-8",
            )

            templates = load_templates(template_dir)

            self.assertIn("debug", templates)
            self.assertEqual(templates["debug"].form["run_name"], "sample")
            self.assertEqual(templates["debug"].overrides, ["model=resnet18"])

    def test_create_run_dir_makes_unique_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            first = create_run_dir(repo_root, "2026-05-08_12-00-00_demo")
            second = create_run_dir(repo_root, "2026-05-08_12-00-00_demo")

            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertNotEqual(first, second)
            self.assertEqual(second.name, "2026-05-08_12-00-00_demo_01")

    def test_status_json_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            run_dir.mkdir()
            status = {"run_id": "demo", "status": "running", "pid": 123}

            write_status(run_dir, status)
            loaded = read_status(run_dir)

            self.assertEqual(loaded, status)

    def test_parse_raw_overrides_ignores_comments_and_blank_lines(self) -> None:
        raw_text = """
        # comment
        model=resnet18

        optimizer=adam
        federated_method=fedavg training_params.batch_size=32
        """

        parsed = parse_raw_overrides(raw_text)

        self.assertEqual(
            parsed,
            [
                "model=resnet18",
                "optimizer=adam",
                "federated_method=fedavg",
                "training_params.batch_size=32",
            ],
        )


if __name__ == "__main__":
    unittest.main()
