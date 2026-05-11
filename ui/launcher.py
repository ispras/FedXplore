from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in bare environments
    yaml = None

RUNS_RELATIVE_DIR = Path("outputs/ui/runs")
DEFAULT_RUN_NAME = "fedxplore_run"
DEFAULT_MLFLOW_REMOTE_URI = "http://10.100.202.109:5000/"
DEFAULT_MLFLOW_AWS_ENDPOINT = "http://10.100.151.14:9000"
DEFAULT_LOCAL_MLFLOW_UI_URL = "http://127.0.0.1:5000/"
STOP_TERM_RETRY_SECONDS = 6
STOP_FORCE_KILL_SECONDS = 18
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
MLFLOW_RUN_URL_PATTERN = re.compile(r"MLFLOW_RUN_URL=(\S+)")
MLFLOW_RUN_ID_PATTERN = re.compile(r"MLFLOW_RUN_ID=(\S+)")
MLFLOW_EXPERIMENT_ID_PATTERN = re.compile(r"MLFLOW_EXPERIMENT_ID=(\S+)")
HYDRA_CONFIG_FILE = "config.yaml"
MLFLOW_UI_REGISTRY_RELATIVE = Path("outputs/ui/mlflow_ui_registry.json")
MLFLOW_UI_LOGS_RELATIVE = Path("outputs/ui/mlflow_ui")
HYDRA_IGNORED_KEYS = {"_target_", "_recursive_", "defaults"}
COMPONENT_FIELD_EXCLUDES: dict[str, set[str]] = {
    "logger": {"run_dir", "run_name"},
    "optimizer": {"params"},
    "preaggregator": {"server"},
}
HYDRA_DEFAULT_MAPPING = {
    "model": "model",
    "logger": "logger",
    "dataset@train_dataset": "train_dataset",
    "dataset@test_dataset": "test_dataset",
    "dataset@trust_dataset": "trust_dataset",
    "distribution": "distribution",
    "model_trainer": "model_trainer",
    "federated_method": "federated_method",
    "client_selector": "client_selector",
    "manager": "manager",
    "losses@loss": "loss",
    "optimizer": "optimizer",
    "preaggregator": "preaggregator",
}


@dataclass(frozen=True)
class TemplateSpec:
    key: str
    path: Path
    name: str
    description: str
    form: dict[str, Any]
    overrides: list[str]


def require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for the FedXplore UI. "
            "Install it with `pip install -r ui/requirements-ui.txt`."
        )


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def seconds_since_iso(value: str | None) -> float | None:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return None
    return max(0.0, (datetime.now().astimezone() - parsed).total_seconds())


def get_repo_root(start_path: Path | None = None) -> Path:
    search_points: list[Path] = []
    if start_path is not None:
        search_points.append(Path(start_path).resolve())
    search_points.append(Path.cwd().resolve())

    for search_point in search_points:
        current = search_point if search_point.is_dir() else search_point.parent
        for candidate in [current, *current.parents]:
            if (candidate / "src/train.py").is_file():
                return candidate

    raise FileNotFoundError(
        "Could not locate the FedXplore repository root. Expected to find src/train.py."
    )


def get_train_entrypoint(repo_root: Path) -> Path:
    train_path = repo_root / "src/train.py"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Could not find train entrypoint at {train_path}. "
            "The UI expects FedXplore/src/train.py to exist."
        )
    return train_path


def get_configs_root(repo_root: Path) -> Path:
    configs_root = repo_root / "src/configs"
    if not configs_root.is_dir():
        raise FileNotFoundError(
            f"Could not find Hydra configs directory at {configs_root}."
        )
    return configs_root


def get_main_config(repo_root: Path) -> dict[str, Any]:
    return read_yaml_file(get_configs_root(repo_root) / HYDRA_CONFIG_FILE)


def get_runs_root(repo_root: Path) -> Path:
    return repo_root / RUNS_RELATIVE_DIR


def sanitize_run_name(name: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in name.strip())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or DEFAULT_RUN_NAME


def preview_stdout_path(repo_root: Path, run_name: str) -> Path:
    return build_user_log_path(repo_root, run_name)


def now_run_id(run_name: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{sanitize_run_name(run_name)}"


def build_user_log_path(repo_root: Path, run_name: str) -> Path:
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir / f"{sanitize_run_name(run_name)}.txt"


def discover_config_group_options(
    repo_root: Path, group_name: str, *, exclude: set[str] | None = None
) -> list[str]:
    config_group_dir = get_configs_root(repo_root) / group_name
    if not config_group_dir.is_dir():
        return []

    excluded = exclude or set()
    options = [
        path.stem
        for path in sorted(config_group_dir.glob("*.yaml"))
        if path.stem not in excluded
    ]
    return options


def normalize_default_option(value: Any) -> str:
    if value in (None, "null"):
        return ""
    return str(value)


def extract_main_default_selections(repo_root: Path) -> dict[str, str]:
    config = get_main_config(repo_root)
    defaults = config.get("defaults", [])
    selections: dict[str, str] = {}
    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for raw_key, raw_value in entry.items():
            mapped_key = HYDRA_DEFAULT_MAPPING.get(raw_key)
            if mapped_key is None:
                continue
            selections[mapped_key] = normalize_default_option(raw_value)

    manager_option = selections.get("manager", "")
    selections["manager_batch_generator"] = extract_manager_batch_generator_default(
        repo_root, manager_option
    )
    return selections


def extract_manager_batch_generator_default(repo_root: Path, manager_option: str) -> str:
    if not manager_option:
        return ""
    manager_config = load_component_config(repo_root, "manager", manager_option)
    defaults = manager_config.get("defaults", [])
    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        if "batch_generator" in entry:
            return normalize_default_option(entry["batch_generator"])
    return ""


def load_component_config(repo_root: Path, group_name: str, option: str | None) -> dict[str, Any]:
    normalized_option = normalize_default_option(option)
    if not normalized_option:
        return {}
    group_path = get_configs_root(repo_root) / group_name
    config_path = group_path / f"{normalized_option}.yaml"
    if not config_path.is_file():
        return {}
    return read_yaml_file(config_path)


def strip_component_config_fields(
    config: dict[str, Any], *, component_name: str
) -> dict[str, Any]:
    excluded_fields = COMPONENT_FIELD_EXCLUDES.get(component_name, set())
    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        if key in HYDRA_IGNORED_KEYS or key in excluded_fields:
            continue
        if isinstance(value, str) and value.strip() == "???":
            continue
        cleaned[key] = value
    return cleaned


def flatten_mapping(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_mapping(value, path))
        else:
            flat[path] = value
    return flat


def unflatten_mapping(flat: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        current = nested
        chunks = [chunk for chunk in path.split(".") if chunk]
        for chunk in chunks[:-1]:
            current = current.setdefault(chunk, {})
        current[chunks[-1]] = value
    return nested


def get_component_default_params(
    repo_root: Path, component_name: str, option: str | None
) -> dict[str, Any]:
    group_name = "losses" if component_name == "loss" else component_name
    if component_name == "manager_batch_generator":
        group_name = "manager/batch_generator"
    if component_name == "attack":
        group_name = "attacks"

    raw_config = load_component_config(repo_root, group_name, option)
    cleaned_config = strip_component_config_fields(
        raw_config,
        component_name=component_name,
    )
    return flatten_mapping(cleaned_config)


def format_hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value == "":
            return "''"
        if all(char.isalnum() or char in "._/-" for char in value):
            return value
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ",".join(format_hydra_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [
            f"{str(key)}:{format_hydra_value(item_value)}"
            for key, item_value in value.items()
        ]
        return "{" + ",".join(items) + "}"
    return json.dumps(value)


def load_templates(template_dir: Path) -> dict[str, TemplateSpec]:
    require_yaml()
    if not template_dir.is_dir():
        return {}

    templates: dict[str, TemplateSpec] = {}
    for template_path in sorted(template_dir.glob("*.yaml")):
        raw = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Template {template_path} must contain a YAML mapping at the top level."
            )

        overrides = raw.get("overrides", [])
        if not isinstance(overrides, list):
            raise ValueError(f"Template {template_path} field 'overrides' must be a list.")

        form = raw.get("form", {})
        if not isinstance(form, dict):
            raise ValueError(f"Template {template_path} field 'form' must be a mapping.")

        key = template_path.stem
        templates[key] = TemplateSpec(
            key=key,
            path=template_path,
            name=str(raw.get("name", key)),
            description=str(raw.get("description", "")).strip(),
            form={str(field): value for field, value in form.items()},
            overrides=[str(override).strip() for override in overrides if str(override).strip()],
        )
    return templates


def read_yaml_file(path: Path) -> dict[str, Any]:
    require_yaml()
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return data


def get_mlflow_defaults(repo_root: Path) -> dict[str, str]:
    try:
        config_values = read_yaml_file(get_configs_root(repo_root) / "logger/mlflow.yaml")
    except RuntimeError:
        config_values = {}
    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    remote_tracking_uri = (
        str(config_values.get("tracking_uri", "")).strip() or DEFAULT_MLFLOW_REMOTE_URI
    )
    tracking_uri = env_tracking_uri or remote_tracking_uri
    experiment_name = str(config_values.get("experiment_name", "")).strip()
    return {
        "remote_tracking_uri": remote_tracking_uri,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "ui_url": normalize_mlflow_ui_url(tracking_uri),
    }


def get_local_mlflow_tracking_uri(repo_root: Path) -> str:
    local_store = (repo_root / "outputs/mlruns").resolve()
    local_store.mkdir(parents=True, exist_ok=True)
    return str(local_store)


def infer_mlflow_target(
    tracking_uri: str | None,
    *,
    remote_tracking_uri: str | None = None,
) -> str:
    candidate = (tracking_uri or "").strip().rstrip("/")
    remote_candidate = (remote_tracking_uri or DEFAULT_MLFLOW_REMOTE_URI).strip().rstrip("/")
    if candidate and candidate == remote_candidate:
        return "remote"
    if candidate.startswith(("http://", "https://")):
        return "remote"
    return "local"


def parse_raw_overrides(text: str) -> list[str]:
    overrides: list[str] = []
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            raise ValueError(f"Could not parse raw override on line {index}: {exc}") from exc
        overrides.extend(part.strip() for part in parts if part.strip())
    return overrides


def normalize_device_ids(device_ids: str) -> str:
    normalized = device_ids.strip()
    if not normalized:
        return ""
    if normalized.startswith("[") and normalized.endswith("]"):
        return normalized.replace(" ", "")

    chunks = [chunk.strip() for chunk in normalized.replace(";", ",").split(",")]
    values = [chunk for chunk in chunks if chunk]
    if not values:
        return ""
    return "[" + ",".join(values) + "]"


def extract_host_from_url(value: str | None) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate)
    return parsed.hostname or ""


def merge_no_proxy_entries(existing_value: str | None, hosts: list[str]) -> str:
    existing_parts = [
        chunk.strip() for chunk in (existing_value or "").split(",") if chunk.strip()
    ]
    merged: list[str] = []
    seen: set[str] = set()
    for entry in [*existing_parts, *hosts]:
        if entry and entry not in seen:
            merged.append(entry)
            seen.add(entry)
    return ",".join(merged)


def build_subprocess_env(
    *,
    disable_proxy: bool = False,
    mlflow_tracking_uri: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    bypass_hosts: list[str] = []

    tracking_host = extract_host_from_url(mlflow_tracking_uri)
    if tracking_host:
        bypass_hosts.append(tracking_host)

    aws_endpoint = env.get("AWS_ENDPOINT_URL") or DEFAULT_MLFLOW_AWS_ENDPOINT
    aws_host = extract_host_from_url(aws_endpoint)
    if aws_host:
        bypass_hosts.append(aws_host)

    if disable_proxy:
        for key in PROXY_ENV_KEYS:
            env.pop(key, None)

    if bypass_hosts:
        env["NO_PROXY"] = merge_no_proxy_entries(env.get("NO_PROXY"), bypass_hosts)
        env["no_proxy"] = merge_no_proxy_entries(env.get("no_proxy"), bypass_hosts)

    return env, bypass_hosts


def get_process_group_id(pid: int | None) -> int | None:
    if pid is None or pid <= 0:
        return None
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def send_signal_to_run(status: dict[str, Any], signum: int) -> bool:
    pgid = status.get("process_group_id")
    if isinstance(pgid, int) and pgid > 0:
        try:
            os.killpg(pgid, signum)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return False

    pid = status.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, signum)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def extract_mlflow_run_url_from_text(text: str) -> str:
    match = MLFLOW_RUN_URL_PATTERN.search(text)
    if match is None:
        return ""
    return match.group(1).strip()


def extract_mlflow_run_id_from_text(text: str) -> str:
    match = MLFLOW_RUN_ID_PATTERN.search(text)
    if match is None:
        return ""
    return match.group(1).strip()


def extract_mlflow_experiment_id_from_text(text: str) -> str:
    match = MLFLOW_EXPERIMENT_ID_PATTERN.search(text)
    if match is None:
        return ""
    return match.group(1).strip()


def build_mlflow_run_url(
    ui_url: str | None,
    experiment_id: str | None,
    run_id: str | None,
) -> str:
    base_url = normalize_mlflow_ui_url(ui_url)
    if not base_url:
        return ""
    if "/#/experiments/" in base_url and "/runs/" in base_url:
        return base_url

    normalized_experiment_id = str(experiment_id or "").strip()
    normalized_run_id = str(run_id or "").strip()
    if not normalized_experiment_id or not normalized_run_id:
        return base_url

    if "/#/experiments" in base_url:
        root = base_url.split("/#/experiments", 1)[0].rstrip("/") + "/#/experiments"
        return f"{root}/{normalized_experiment_id}/runs/{normalized_run_id}"
    if base_url.startswith(("http://", "https://")):
        return (
            base_url.rstrip("/")
            + f"/#/experiments/{normalized_experiment_id}/runs/{normalized_run_id}"
        )
    return base_url


def persist_mlflow_metadata(
    run_dir: Path,
    status: dict[str, Any],
    *,
    mlflow_url: str | None = None,
    mlflow_run_id: str | None = None,
    mlflow_experiment_id: str | None = None,
) -> dict[str, Any]:
    updated_status = dict(status)
    changed = False

    metadata_updates = {
        "mlflow_url": str(mlflow_url or "").strip(),
        "mlflow_run_id": str(mlflow_run_id or "").strip(),
        "mlflow_experiment_id": str(mlflow_experiment_id or "").strip(),
    }
    for key, value in metadata_updates.items():
        if not value:
            continue
        if updated_status.get(key) != value:
            updated_status[key] = value
            changed = True

    if metadata_updates["mlflow_url"]:
        (run_dir / "mlflow_url.txt").write_text(
            metadata_updates["mlflow_url"],
            encoding="utf-8",
        )

    if changed:
        write_status(run_dir, updated_status)

    spec = read_spec(run_dir)
    if spec:
        spec_changed = False
        for key, value in metadata_updates.items():
            if not value:
                continue
            if spec.get(key) != value:
                spec[key] = value
                spec_changed = True
        if spec_changed:
            write_yaml_or_json(run_dir / "spec.yaml", spec)

    return updated_status


def update_mlflow_url_from_log(run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    stdout_path = Path(str(status.get("stdout_path", "") or ""))
    if not stdout_path.is_file():
        return status

    try:
        log_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return status

    explicit_url = extract_mlflow_run_url_from_text(log_text)
    run_id = extract_mlflow_run_id_from_text(log_text)
    experiment_id = extract_mlflow_experiment_id_from_text(log_text)

    mlflow_url = explicit_url
    if not mlflow_url:
        mlflow_url = build_mlflow_run_url(
            status.get("mlflow_url"),
            experiment_id,
            run_id,
        )

    if (
        not mlflow_url
        and not run_id
        and not experiment_id
    ):
        return status

    current_url = str(status.get("mlflow_url", "") or "").strip()
    if (
        mlflow_url == current_url
        and run_id == str(status.get("mlflow_run_id", "") or "").strip()
        and experiment_id == str(status.get("mlflow_experiment_id", "") or "").strip()
    ):
        return status

    return persist_mlflow_metadata(
        run_dir,
        status,
        mlflow_url=mlflow_url or None,
        mlflow_run_id=run_id or None,
        mlflow_experiment_id=experiment_id or None,
    )


def extract_override_key(override: str) -> str:
    cleaned = override.strip()
    while cleaned.startswith(("+", "~")):
        cleaned = cleaned[1:]
    return cleaned.split("=", 1)[0].strip()


def find_duplicate_override_keys(overrides: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for override in overrides:
        key = extract_override_key(override)
        if not key:
            continue
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def build_overrides(form_values: dict[str, Any], raw_overrides: list[str]) -> list[str]:
    if "selected_groups" in form_values:
        overrides: list[str] = []
        selected_groups: dict[str, str] = form_values["selected_groups"]
        base_params: dict[str, Any] = form_values.get("base_params", {})
        component_params: dict[str, dict[str, Any]] = form_values.get("component_params", {})

        selection_override_map = [
            ("model", "model"),
            ("logger", "logger"),
            ("train_dataset", "dataset@train_dataset"),
            ("test_dataset", "dataset@test_dataset"),
            ("trust_dataset", "dataset@trust_dataset"),
            ("distribution", "distribution"),
            ("model_trainer", "model_trainer"),
            ("federated_method", "federated_method"),
            ("client_selector", "client_selector"),
            ("manager", "manager"),
            ("loss", "losses@loss"),
            ("optimizer", "optimizer"),
            ("preaggregator", "preaggregator"),
            ("manager_batch_generator", "manager/batch_generator"),
        ]
        optional_group_keys = {"trust_dataset", "preaggregator"}
        for selection_key, hydra_key in selection_override_map:
            if selection_key not in selected_groups:
                continue
            selected_option = normalize_default_option(selected_groups[selection_key])
            if selection_key in optional_group_keys and not selected_option:
                continue
            if not selected_option:
                continue
            overrides.append(f"{hydra_key}={selected_option}")

        for path, value in base_params.items():
            overrides.append(f"{path}={format_hydra_value(value)}")

        component_prefix_map = {
            "distribution": "distribution",
            "model": "model",
            "model_trainer": "model_trainer",
            "federated_method": "federated_method",
            "client_selector": "client_selector",
            "logger": "logger",
            "optimizer": "optimizer",
            "loss": "loss",
            "manager": "manager",
            "manager_batch_generator": "manager.batch_generator",
            "preaggregator": "preaggregator",
        }
        for component_name, prefix in component_prefix_map.items():
            selected_option = normalize_default_option(selected_groups.get(component_name, ""))
            if component_name in {"preaggregator", "trust_dataset"} and not selected_option:
                continue
            if component_name == "logger" and selected_option == "base":
                # Base logger only needs group selection unless the user edited extra params.
                pass
            for path, value in component_params.get(component_name, {}).items():
                overrides.append(f"{prefix}.{path}={format_hydra_value(value)}")

        if selected_groups.get("logger") == "mlflow":
            run_name = str(form_values.get("run_name", "")).strip()
            if run_name:
                overrides.append(f"logger.run_name={format_hydra_value(run_name)}")

        overrides.extend(raw_overrides)
        return overrides

    overrides: list[str] = [
        f"random_state={int(form_values['random_state'])}",
        f"federated_params.communication_rounds={int(form_values['communication_rounds'])}",
        f"federated_params.amount_of_clients={int(form_values['amount_of_clients'])}",
        f"federated_params.client_subset_size={int(form_values['client_subset_size'])}",
        f"training_params.batch_size={int(form_values['training_batch_size'])}",
        f"manager.batch_generator.batch_size={int(form_values['manager_batch_size'])}",
        f"federated_params.print_client_metrics={bool(form_values['print_client_metrics'])}",
        f"dataset@train_dataset={form_values['dataset']}",
        f"dataset@test_dataset={form_values['dataset']}",
        f"distribution={form_values['distribution']}",
        f"federated_method={form_values['federated_method']}",
        f"client_selector={form_values['client_selector']}",
        f"logger={form_values['logger']}",
    ]

    trust_dataset = str(form_values.get("trust_dataset", "")).strip()
    if trust_dataset:
        overrides.append(f"dataset@trust_dataset={trust_dataset}")

    distribution = str(form_values["distribution"])
    if distribution.startswith("dirichlet"):
        overrides.append(f"distribution.alpha={form_values['distribution_alpha']}")

    device_mode = str(form_values["device_mode"])
    overrides.append(f"training_params.device={'cpu' if device_mode == 'cpu' else 'cuda'}")
    if device_mode == "gpu":
        device_ids = normalize_device_ids(str(form_values.get("device_ids", "")))
        if device_ids:
            overrides.append(f"training_params.device_ids={device_ids}")

    if str(form_values["logger"]) == "mlflow":
        tracking_uri = str(form_values.get("mlflow_tracking_uri", "")).strip()
        experiment_name = str(form_values.get("mlflow_experiment_name", "")).strip()
        if tracking_uri:
            overrides.append(f"logger.tracking_uri={tracking_uri}")
        if experiment_name:
            overrides.append(f"logger.experiment_name={experiment_name}")

        run_name = str(form_values.get("run_name", "")).strip()
        if run_name:
            overrides.append(f"logger.run_name={run_name}")

    overrides.extend(raw_overrides)
    return overrides


def build_command(repo_root: Path, overrides: list[str]) -> list[str]:
    train_entrypoint = get_train_entrypoint(repo_root)
    relative_entrypoint = train_entrypoint.relative_to(repo_root).as_posix()
    return [sys.executable, relative_entrypoint, *overrides]


def format_shell_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def format_manual_shell_command(cmd: list[str], stdout_path: Path) -> str:
    return (
        format_shell_command(cmd)
        + " > "
        + shlex.quote(str(stdout_path))
        + " 2>&1"
    )


def create_run_dir(repo_root: Path, run_id: str) -> Path:
    runs_root = get_runs_root(repo_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    candidate = runs_root / run_id
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 1
    while True:
        alternative = runs_root / f"{run_id}_{suffix:02d}"
        if not alternative.exists():
            alternative.mkdir(parents=True, exist_ok=False)
            return alternative
        suffix += 1


def write_yaml_or_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return
    if path.suffix in {".yaml", ".yml"}:
        require_yaml()
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        return
    raise ValueError(f"Unsupported file extension for {path}.")


def read_spec(run_dir: Path) -> dict[str, Any]:
    spec_path = run_dir / "spec.yaml"
    if not spec_path.is_file():
        return {}
    return read_yaml_file(spec_path)


def read_status(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return {
            "run_id": run_dir.name,
            "run_name": run_dir.name,
            "status": "missing_status",
        }

    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "run_id": run_dir.name,
            "run_name": run_dir.name,
            "status": "invalid_status",
            "error": str(exc),
        }


def write_status(run_dir: Path, status: dict[str, Any]) -> None:
    write_yaml_or_json(run_dir / "status.json", status)


def append_run_event(
    run_dir: Path,
    event_type: str,
    *,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    event = {
        "timestamp": iso_now(),
        "event_type": event_type,
        "message": message or event_type,
        "payload": payload or {},
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def read_run_events(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def is_pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    proc_stat_path = Path("/proc") / str(pid) / "stat"
    if proc_stat_path.is_file():
        try:
            stat_chunks = proc_stat_path.read_text(encoding="utf-8").split()
        except OSError:
            stat_chunks = []
        if len(stat_chunks) >= 3 and stat_chunks[2] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refresh_run_status(run_dir: Path, status: dict[str, Any] | None = None) -> dict[str, Any]:
    status = dict(status or read_status(run_dir))
    pid = status.get("pid")
    current_status = status.get("status", "unknown")

    if current_status == "unknown_finished":
        status["status"] = "finished"
        current_status = "finished"
        write_status(run_dir, status)

    if current_status in {"running", "stopping"} and isinstance(pid, int) and not is_pid_alive(pid):
        status["status"] = "stopped" if current_status == "stopping" else "finished"
        if not status.get("finished_at"):
            status["finished_at"] = iso_now()
        write_status(run_dir, status)
        append_run_event(
            run_dir,
            "status_changed",
            message=f"Status changed to {status['status']}",
            payload={"status": status["status"]},
        )

    if status.get("status") == "stopping" and isinstance(pid, int) and is_pid_alive(pid):
        stopping_for = seconds_since_iso(status.get("stop_requested_at"))
        last_signal_for = seconds_since_iso(status.get("last_signal_at"))
        signal_stage = str(status.get("stop_signal_stage", "term"))

        if stopping_for is not None and stopping_for >= STOP_FORCE_KILL_SECONDS and signal_stage != "kill":
            if send_signal_to_run(status, signal.SIGKILL):
                status["stop_signal_stage"] = "kill"
                status["last_signal"] = "SIGKILL"
                status["last_signal_at"] = iso_now()
                write_status(run_dir, status)
                append_run_event(
                    run_dir,
                    "force_kill_requested",
                    message="SIGKILL sent after stop timeout",
                    payload={"pid": pid, "process_group_id": status.get("process_group_id")},
                )
        elif (
            stopping_for is not None
            and stopping_for >= STOP_TERM_RETRY_SECONDS
            and (last_signal_for is None or last_signal_for >= STOP_TERM_RETRY_SECONDS)
            and signal_stage != "kill"
        ):
            if send_signal_to_run(status, signal.SIGTERM):
                status["stop_signal_stage"] = "term"
                status["last_signal"] = "SIGTERM"
                status["last_signal_at"] = iso_now()
                write_status(run_dir, status)
                append_run_event(
                    run_dir,
                    "stop_retried",
                    message="SIGTERM resent to process group",
                    payload={"pid": pid, "process_group_id": status.get("process_group_id")},
                )

    return update_mlflow_url_from_log(run_dir, status)


def render_command_script(repo_root: Path, command: str) -> str:
    return (
        "#!/usr/bin/env bash\n"
        f"cd {shlex.quote(str(repo_root))}\n"
        f"exec {command}\n"
    )


def start_run(
    repo_root: Path,
    run_name: str,
    overrides: list[str],
    mlflow_url: str | None = None,
    spec_data: dict[str, Any] | None = None,
    subprocess_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    base_run_id = now_run_id(run_name)
    run_dir = create_run_dir(repo_root, base_run_id)
    run_id = run_dir.name
    cmd = build_command(repo_root, overrides)
    stdout_path = build_user_log_path(repo_root, run_name)
    stderr_path = stdout_path
    shell_command = format_manual_shell_command(cmd, stdout_path)
    created_at = iso_now()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    spec_payload = {
        "run_id": run_id,
        "run_name": run_name,
        "created_at": created_at,
        "repo_root": str(repo_root),
        "cwd": str(repo_root),
        "argv": cmd,
        "command": shell_command,
        "overrides": overrides,
        "mlflow_url": mlflow_url,
        "output_log_path": str(stdout_path),
    }
    if spec_data:
        spec_payload.update(spec_data)

    write_yaml_or_json(run_dir / "spec.yaml", spec_payload)

    command_script = render_command_script(repo_root, shell_command)
    command_path = run_dir / "command.sh"
    command_path.write_text(command_script, encoding="utf-8")
    command_path.chmod(0o755)

    if mlflow_url:
        (run_dir / "mlflow_url.txt").write_text(mlflow_url, encoding="utf-8")

    run_stdout_link = run_dir / "stdout.txt"
    if run_stdout_link.exists() or run_stdout_link.is_symlink():
        run_stdout_link.unlink()
    run_stdout_link.symlink_to(stdout_path)

    stdout_handle = stdout_path.open("w", encoding="utf-8", buffering=1)
    try:
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=stdout_handle,
            stderr=stdout_handle,
            text=True,
            start_new_session=True,
            env=subprocess_env,
        )
    except Exception as exc:
        stdout_handle.close()
        status = {
            "run_id": run_id,
            "run_name": run_name,
            "status": "failed_to_start",
            "pid": None,
            "created_at": created_at,
            "started_at": None,
            "finished_at": iso_now(),
            "returncode": None,
            "command": shell_command,
            "cwd": str(repo_root),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "mlflow_url": mlflow_url,
            "error": str(exc),
        }
        write_status(run_dir, status)
        append_run_event(
            run_dir,
            "failed_to_start",
            message=str(exc),
            payload={"error": str(exc)},
        )
        raise RuntimeError(f"Failed to start run {run_id}: {exc}") from exc

    stdout_handle.close()

    process_group_id = get_process_group_id(process.pid)
    (run_dir / "pid.txt").write_text(f"{process.pid}\n", encoding="utf-8")

    status = {
        "run_id": run_id,
        "run_name": run_name,
        "status": "running",
        "pid": process.pid,
        "created_at": created_at,
        "started_at": created_at,
        "finished_at": None,
        "returncode": None,
        "command": shell_command,
        "cwd": str(repo_root),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "mlflow_url": mlflow_url,
        "process_group_id": process_group_id,
    }
    write_status(run_dir, status)
    append_run_event(
        run_dir,
        "started",
        message=f"Run started with pid {process.pid}",
        payload={"pid": process.pid, "process_group_id": process_group_id},
    )
    return status


def stop_run(run_dir: Path) -> dict[str, Any]:
    status = refresh_run_status(run_dir)
    pid = status.get("pid")

    if status.get("status") not in {"running", "stopping"}:
        return status

    if not isinstance(pid, int) or pid <= 0:
        status["status"] = "missing_pid"
        if not status.get("finished_at"):
            status["finished_at"] = iso_now()
        write_status(run_dir, status)
        return status

    if not is_pid_alive(pid):
        status["status"] = "finished"
        if not status.get("finished_at"):
            status["finished_at"] = iso_now()
        write_status(run_dir, status)
        return status

    if not send_signal_to_run(status, signal.SIGTERM):
        status["status"] = "finished"
        if not status.get("finished_at"):
            status["finished_at"] = iso_now()
        write_status(run_dir, status)
        return status

    status["status"] = "stopping"
    status["stop_requested_at"] = iso_now()
    status["last_signal"] = "SIGTERM"
    status["last_signal_at"] = status["stop_requested_at"]
    status["stop_signal_stage"] = "term"
    write_status(run_dir, status)
    append_run_event(
        run_dir,
        "stop_requested",
        message="SIGTERM sent to process group",
        payload={"pid": pid, "process_group_id": status.get("process_group_id")},
    )

    for _ in range(4):
        time.sleep(0.5)
        if not is_pid_alive(pid):
            status["status"] = "stopped"
            status["finished_at"] = iso_now()
            write_status(run_dir, status)
            append_run_event(
                run_dir,
                "status_changed",
                message="Status changed to stopped",
                payload={"status": "stopped"},
            )
            return status

    send_signal_to_run(status, signal.SIGTERM)
    status["last_signal_at"] = iso_now()
    write_status(run_dir, status)
    return status


def list_runs(repo_root: Path) -> list[dict[str, Any]]:
    runs_root = get_runs_root(repo_root)
    if not runs_root.exists():
        runs_root.mkdir(parents=True, exist_ok=True)
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        status = refresh_run_status(run_dir)
        status.setdefault("run_id", run_dir.name)
        status.setdefault("run_name", run_dir.name)
        status["run_dir"] = str(run_dir)
        runs.append(status)
    return runs


def tail_file(path: Path, n_lines: int) -> str:
    if not path.is_file():
        return f"File not found: {path}"

    last_lines: deque[str] = deque(maxlen=max(1, n_lines))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            last_lines.append(line.rstrip("\n"))
    return "\n".join(last_lines)


def normalize_mlflow_ui_url(url: str | None) -> str:
    candidate = (url or "").strip()
    if not candidate:
        return ""
    if candidate.startswith(("http://", "https://")) and "#/" not in candidate:
        return candidate.rstrip("/") + "/#/experiments"
    return candidate


def parse_mlflow_ui_binding(ui_url: str | None) -> tuple[str, str, int]:
    normalized_url = normalize_mlflow_ui_url(ui_url or DEFAULT_LOCAL_MLFLOW_UI_URL)
    parsed = urlparse(normalized_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    default_port = 443 if scheme == "https" else 5000
    port = parsed.port or default_port
    return scheme, host, port


def canonicalize_local_tracking_uri(tracking_uri: str) -> Path:
    candidate = str(tracking_uri or "").strip()
    if not candidate:
        raise RuntimeError("Local MLflow tracking URI is empty.")
    parsed = urlparse(candidate)
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    if parsed.scheme in {"", None}:
        return Path(candidate).expanduser().resolve()
    raise RuntimeError(
        f"Local MLflow auto-launch supports only filesystem tracking URIs. Got: {tracking_uri}"
    )


def get_mlflow_ui_registry_path(repo_root: Path) -> Path:
    return repo_root / MLFLOW_UI_REGISTRY_RELATIVE


def read_mlflow_ui_registry(repo_root: Path) -> dict[str, dict[str, Any]]:
    registry_path = get_mlflow_ui_registry_path(repo_root)
    if not registry_path.is_file():
        return {}
    try:
        raw_data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw_data, dict):
        return {}

    registry: dict[str, dict[str, Any]] = {}
    changed = False
    for store_path, record in raw_data.items():
        if not isinstance(record, dict):
            changed = True
            continue
        pid = record.get("pid")
        if isinstance(pid, int) and not is_pid_alive(pid):
            changed = True
            continue
        registry[str(store_path)] = record

    if changed:
        write_mlflow_ui_registry(repo_root, registry)
    return registry


def write_mlflow_ui_registry(repo_root: Path, registry: dict[str, dict[str, Any]]) -> None:
    registry_path = get_mlflow_ui_registry_path(repo_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def is_tcp_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_tcp_listener(host: str, port: int, timeout_seconds: float = 8.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.4)
            try:
                sock.connect((host, port))
            except OSError:
                time.sleep(0.2)
                continue
        return True
    return False


def pick_available_port(host: str, preferred_port: int) -> int:
    for candidate in range(preferred_port, preferred_port + 50):
        if is_tcp_port_available(host, candidate):
            return candidate
    raise RuntimeError(
        f"Could not find a free TCP port for local MLflow UI near {host}:{preferred_port}."
    )


def get_local_mlflow_ui_log_path(repo_root: Path, store_path: Path) -> Path:
    digest = hashlib.sha1(str(store_path).encode("utf-8")).hexdigest()[:12]
    logs_dir = repo_root / MLFLOW_UI_LOGS_RELATIVE
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{digest}.log"


def ensure_local_mlflow_ui(
    repo_root: Path,
    tracking_uri: str,
    *,
    preferred_ui_url: str | None = None,
) -> str:
    store_path = canonicalize_local_tracking_uri(tracking_uri)
    store_path.mkdir(parents=True, exist_ok=True)
    store_key = str(store_path)
    registry = read_mlflow_ui_registry(repo_root)
    existing_record = registry.get(store_key, {})
    existing_pid = existing_record.get("pid")
    existing_url = normalize_mlflow_ui_url(existing_record.get("url"))
    if isinstance(existing_pid, int) and is_pid_alive(existing_pid) and existing_url:
        return existing_url

    scheme, host, preferred_port = parse_mlflow_ui_binding(preferred_ui_url)
    selected_port = preferred_port
    if not is_tcp_port_available(host, selected_port):
        selected_port = pick_available_port(host, preferred_port + 1)

    log_path = get_local_mlflow_ui_log_path(repo_root, store_path)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                str(store_path),
                "--host",
                host,
                "--port",
                str(selected_port),
            ],
            cwd=repo_root,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            start_new_session=True,
            env=os.environ.copy(),
        )

    ui_url = f"{scheme}://{host}:{selected_port}/#/experiments"
    if not wait_for_tcp_listener(host, selected_port, timeout_seconds=8.0):
        process.poll()
        log_excerpt = tail_file(log_path, 40) if log_path.is_file() else ""
        raise RuntimeError(
            "Failed to start local MLflow UI. "
            f"Checked {ui_url}. See {log_path} for details.\n{log_excerpt}"
        )

    registry[store_key] = {
        "pid": process.pid,
        "url": ui_url,
        "store_path": store_key,
        "log_path": str(log_path),
        "started_at": iso_now(),
    }
    write_mlflow_ui_registry(repo_root, registry)
    return ui_url


def query_gpus() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    reader = csv.reader(StringIO(result.stdout))
    gpus: list[dict[str, Any]] = []
    for row in reader:
        if len(row) != 5:
            continue
        index, name, memory_used, memory_total, utilization = [cell.strip() for cell in row]
        gpus.append(
            {
                "index": int(index) if index.isdigit() else index,
                "name": name,
                "memory used MiB": int(memory_used) if memory_used.isdigit() else memory_used,
                "memory total MiB": int(memory_total) if memory_total.isdigit() else memory_total,
                "utilization %": int(utilization) if utilization.isdigit() else utilization,
            }
        )
    return gpus
