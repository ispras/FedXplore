from __future__ import annotations

import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ui.launcher import (
    build_rerun_request,
    read_spec,
    start_run,
    write_yaml_or_json,
)
from ui.provenance import PROVENANCE_FILE_NAME, load_provenance


@unittest.skipUnless(find_spec("yaml") is not None, "PyYAML is not installed")
class RerunLauncherTests(unittest.TestCase):
    def make_repo_root(self, tmp_path: Path) -> Path:
        repo_root = tmp_path / "repo"
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/train.py").write_text("print('ok')\n", encoding="utf-8")
        return repo_root

    def test_build_rerun_request_preserves_configuration_and_refreshes_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_run_dir = Path(tmp_dir) / "source-run"
            source_run_dir.mkdir()
            source_overrides = [
                "model=resnet18",
                "federated_method=fedavg",
                "training_params.batch_size=64",
                "logger=mlflow",
                "logger.tracking_uri=http://mlflow.example.test:5000",
                "logger.experiment_name=baseline",
                "logger.run_name=old_mlflow_identity",
            ]
            write_yaml_or_json(
                source_run_dir / "spec.yaml",
                {
                    "run_id": "source-run-789461",
                    "run_name": "Baseline run",
                    "overrides": source_overrides,
                    "mlflow_url": (
                        "http://mlflow.example.test:5000/"
                        "#/experiments/42/runs/old-run"
                    ),
                    "mlflow_run_id": "old-run",
                    "mlflow_experiment_id": "42",
                    "form_payload": {
                        "run_name": "Baseline run",
                        "selected_groups": {"logger": "mlflow"},
                        "component_params": {
                            "logger": {
                                "tracking_uri": "http://mlflow.example.test:5000",
                                "experiment_name": "baseline",
                            }
                        },
                    },
                },
            )

            fresh_run_name = "baseline_run_rerun_20260914_120000_000001"
            with (
                patch.dict(
                    os.environ,
                    {
                        "HTTP_PROXY": "http://proxy.example.test:8080",
                        "HTTPS_PROXY": "http://proxy.example.test:8080",
                        "NO_PROXY": "existing.example.test",
                    },
                    clear=True,
                ),
                patch(
                    "ui.launcher.make_rerun_name", return_value=fresh_run_name
                ) as make_name,
            ):
                request = build_rerun_request(source_run_dir)

            make_name.assert_called_once_with("Baseline run")
            self.assertEqual(request["run_name"], fresh_run_name)
            self.assertNotEqual(request["run_name"], "Baseline run")
            self.assertEqual(
                request["overrides"],
                [
                    *[
                        override
                        for override in source_overrides
                        if override != "logger.run_name=old_mlflow_identity"
                    ],
                    f"logger.run_name={fresh_run_name}",
                ],
            )
            self.assertNotIn("logger.run_name=old_mlflow_identity", request["overrides"])
            self.assertIsNone(request["mlflow_url"])

            spec_data = request["spec_data"]
            self.assertEqual(spec_data["rerun_of"], "source-run-789461")
            self.assertEqual(spec_data["rerun_source_name"], "Baseline run")
            self.assertEqual(spec_data["form_payload"]["run_name"], fresh_run_name)
            self.assertNotIn("mlflow_url", spec_data)
            self.assertNotIn("mlflow_run_id", spec_data)
            self.assertNotIn("mlflow_experiment_id", spec_data)

            subprocess_env = request["subprocess_env"]
            self.assertIsNotNone(subprocess_env)
            assert subprocess_env is not None  # Narrows the value for type checkers.
            self.assertNotIn("HTTP_PROXY", subprocess_env)
            self.assertNotIn("HTTPS_PROXY", subprocess_env)
            self.assertIn("existing.example.test", subprocess_env["NO_PROXY"])
            self.assertIn("mlflow.example.test", subprocess_env["NO_PROXY"])
            self.assertIn("mlflow.example.test", subprocess_env["no_proxy"])
            self.assertEqual(spec_data["proxy_bypass_hosts"], ["mlflow.example.test"])

    def test_start_run_persists_provenance_before_starting_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = self.make_repo_root(Path(tmp_dir))
            observed_run_dirs: list[Path] = []

            def fake_popen(*args: object, **kwargs: object) -> SimpleNamespace:
                runs_root = repo_root / "outputs/ui/runs"
                run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
                self.assertEqual(len(run_dirs), 1)
                run_dir = run_dirs[0]
                observed_run_dirs.append(run_dir)
                self.assertTrue(
                    (run_dir / PROVENANCE_FILE_NAME).is_file(),
                    "provenance must be persisted before the training process starts",
                )
                self.assertTrue((run_dir / "spec.yaml").is_file())
                self.assertEqual(kwargs["cwd"], repo_root)
                return SimpleNamespace(pid=43210)

            # Replace only the launcher module's subprocess reference: the real
            # provenance helper may still use subprocess.run to inspect Git.
            fake_subprocess = SimpleNamespace(Popen=fake_popen)
            with (
                patch("ui.launcher.subprocess", fake_subprocess),
                patch("ui.launcher.get_process_group_id", return_value=43210),
            ):
                status = start_run(
                    repo_root,
                    "provenance check",
                    ["logger=base", "training_params.batch_size=8"],
                    spec_data={"rerun_of": "source-run-789461"},
                )

            self.assertEqual(len(observed_run_dirs), 1)
            run_dir = observed_run_dirs[0]
            self.assertEqual(run_dir.name, status["run_id"])
            self.assertEqual(status["status"], "running")
            self.assertEqual(status["rerun_of"], "source-run-789461")
            self.assertTrue((run_dir / PROVENANCE_FILE_NAME).is_file())
            self.assertIsNotNone(load_provenance(run_dir))
            self.assertEqual(read_spec(run_dir)["rerun_of"], "source-run-789461")


if __name__ == "__main__":
    unittest.main()
