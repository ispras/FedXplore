"""Reusable configuration comparison helpers for the research workbench UI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


# These fields describe execution/storage rather than the experiment that was
# configured.  They are intentionally excluded from a comparison by default.
VOLATILE_SPEC_FIELDS = {
    "argv",
    "command",
    "created_at",
    "cwd",
    "mlflow_experiment_id",
    "mlflow_url",
    "output_log_path",
    "provenance_capture_error",
    "repo_root",
    "run_id",
    "run_name",
    "ui_state_snapshot",
    "proxy_bypass_hosts",
    "rerun_of",
    "rerun_source_name",
}


@dataclass(frozen=True)
class ConfigDiffRow:
    parameter: str
    values: dict[str, str]
    differs: bool


def stable_value(value: Any) -> str:
    """Render lists and complex values deterministically for a comparison."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def flatten_config(data: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten arbitrary nested mappings into dotted paths.

    Lists and values other than mappings remain one stable cell.  This avoids
    inventing fragile index paths while retaining a useful generic diff for
    Hydra configurations and legacy saved specifications alike.
    """

    flattened: dict[str, str] = {}
    for raw_key, value in data.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_config(value, path))
        else:
            flattened[path] = stable_value(value)
    return flattened


def experiment_config_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the persisted experiment definition while omitting run metadata.

    Modern UI runs have a ``form_payload`` that precisely describes their
    configured groups and parameters.  Legacy runs fall back to a sanitized
    saved specification rather than guessing fields from the current source
    tree.
    """

    overrides = spec.get("overrides")
    if isinstance(overrides, list) and overrides:
        # The final override list is the closest saved representation of what
        # Hydra actually received: it includes template and raw overrides and
        # also works for older UI runs that predate form_payload.
        rendered_overrides: dict[str, Any] = {}
        for raw_override in overrides:
            override = str(raw_override).strip()
            while override.startswith(("+", "~")):
                override = override[1:]
            if "=" not in override:
                continue
            key, value = override.split("=", 1)
            key = key.strip()
            if key:
                # Hydra resolves duplicate keys from left to right, so the
                # last stored value is the one that matters for comparison.
                rendered_overrides[key] = value.strip()
        if rendered_overrides:
            return rendered_overrides

    form_payload = spec.get("form_payload")
    if isinstance(form_payload, Mapping):
        return dict(form_payload)
    return {
        str(key): value
        for key, value in spec.items()
        if str(key) not in VOLATILE_SPEC_FIELDS
    }


def build_config_diff(
    configs_by_run: Mapping[str, Mapping[str, Any]],
) -> list[ConfigDiffRow]:
    """Build generic N-run diff rows, including missing values as ``N/A``."""

    flattened_by_run = {
        str(run_id): flatten_config(config)
        for run_id, config in configs_by_run.items()
    }
    all_paths = sorted(
        {
            path
            for flattened in flattened_by_run.values()
            for path in flattened
        }
    )
    rows: list[ConfigDiffRow] = []
    for path in all_paths:
        values = {
            run_id: flattened.get(path, "N/A")
            for run_id, flattened in flattened_by_run.items()
        }
        rows.append(
            ConfigDiffRow(
                parameter=path,
                values=values,
                differs=len(set(values.values())) > 1,
            )
        )
    return rows
