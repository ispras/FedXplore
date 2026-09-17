"""Curated, reproducible multi-run Examples for the FedXplore UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class ExampleDefinition:
    key: str
    data: dict[str, Any]

    @property
    def title(self) -> str:
        return str(self.data["title"])

    @property
    def preferred_metrics(self) -> list[dict[str, str]]:
        return [dict(item) for item in self.data.get("preferred_metrics", [])]


@dataclass(frozen=True)
class ExampleLaunchRequest:
    example_key: str
    group_id: str
    condition: str
    display_label: str
    run_name: str
    method: str
    overrides: list[str]
    subprocess_env: dict[str, str]
    spec_data: dict[str, Any]


def load_examples(path: Path) -> dict[str, ExampleDefinition]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("examples", {})
    if not isinstance(entries, dict):
        raise ValueError("examples.yaml must contain an 'examples' mapping")
    definitions: dict[str, ExampleDefinition] = {}
    required = {"title", "poster", "group_prefix", "runs", "common_overrides", "preferred_metrics"}
    for key, value in entries.items():
        if not isinstance(value, dict) or not required <= set(value):
            raise ValueError(f"Example {key!r} is incomplete")
        if not isinstance(value["runs"], list) or not value["runs"]:
            raise ValueError(f"Example {key!r} must contain at least one run")
        definitions[str(key)] = ExampleDefinition(str(key), dict(value))
    return definitions


def ordered_examples(examples: Mapping[str, ExampleDefinition]) -> list[ExampleDefinition]:
    return sorted(examples.values(), key=lambda item: (int(item.data.get("order", 10_000)), item.key))


def build_group_id(example: ExampleDefinition, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{example.data['group_prefix']}_{timestamp}"


def execution_overrides(device: str, gpu_ids: list[int], seed: int) -> list[str]:
    normalized_device = "cuda" if device == "cuda" else "cpu"
    ids = [int(item) for item in gpu_ids] if normalized_device == "cuda" else []
    return [
        f"random_state={int(seed)}",
        f"training_params.device={normalized_device}",
        "training_params.device_ids=[" + ",".join(str(item) for item in ids) + "]",
    ]


def build_example_launch_plan(
    example: ExampleDefinition,
    *,
    group_id: str,
    device: str,
    gpu_ids: list[int],
    seed: int,
    tracking_uri: str,
    base_env: Mapping[str, str],
) -> list[ExampleLaunchRequest]:
    """Build exact script-equivalent requests without starting processes."""

    env = dict(base_env)
    env.update(THREAD_LIMITS)
    common = [str(item) for item in example.data["common_overrides"]]
    dynamic = execution_overrides(device, gpu_ids, seed)
    run_specs = list(example.data["runs"])
    plan: list[ExampleLaunchRequest] = []
    for index, raw_run in enumerate(run_specs, start=1):
        if not isinstance(raw_run, dict):
            raise ValueError(f"Example {example.key!r} has an invalid run entry")
        condition = str(raw_run["condition"])
        method = str(raw_run["method"])
        label = str(raw_run["label"])
        logger_overrides = [
            f"logger.tracking_uri={tracking_uri}",
            f"logger.experiment_name={example.data['mlflow_experiment']}",
            f"logger.run_name={group_id}/{condition}",
            f"+logger.tags.toy_suite={example.key}",
            f"+logger.tags.run_group={group_id}",
            f"+logger.tags.condition={condition}",
            f"+logger.tags.method={method}",
            f"+logger.tags.seed={int(seed)}",
            "+logger.tags.source=ui_examples",
            f"hydra.run.dir={example.data['hydra_output_root']}/{group_id}/{condition}_hydra",
        ]
        metadata = {
            "example_batch": {
                "schema_version": 1,
                "example_key": example.key,
                "example_title": example.title,
                "group_id": group_id,
                "condition": condition,
                "run_label": label,
                "run_index": index,
                "run_count": len(run_specs),
                "preferred_metrics": [item["metric"] for item in example.preferred_metrics],
            }
        }
        plan.append(
            ExampleLaunchRequest(
                example_key=example.key,
                group_id=group_id,
                condition=condition,
                display_label=label,
                run_name=str(raw_run["run_name"]),
                method=method,
                overrides=[*common, *dynamic, *(str(item) for item in raw_run.get("overrides", [])), *logger_overrides],
                subprocess_env=dict(env),
                spec_data=metadata,
            )
        )
    return plan


def launch_example_suite(
    plan: list[ExampleLaunchRequest],
    launch: Callable[..., dict[str, Any]],
    *,
    repo_root: Path,
    mlflow_url: str,
    on_attempt: Callable[[ExampleLaunchRequest, dict[str, Any] | None, Exception | None], None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Start every child, retaining successes when a later launch fails."""

    statuses: list[dict[str, Any]] = []
    errors: list[str] = []
    for request in plan:
        try:
            status = launch(
                    repo_root,
                    request.run_name,
                    request.overrides,
                    mlflow_url=mlflow_url,
                    spec_data=request.spec_data,
                    subprocess_env=request.subprocess_env,
            )
            statuses.append(status)
            if on_attempt:
                on_attempt(request, status, None)
        except Exception as exc:  # Each later child must still be attempted.
            errors.append(f"{request.display_label}: {exc}")
            if on_attempt:
                on_attempt(request, None, exc)
    return statuses, errors


def example_context_from_specs(specs: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Recover durable Example context only for one homogeneous selected batch."""

    batches = [spec.get("example_batch") for spec in specs]
    if not batches or any(not isinstance(batch, Mapping) for batch in batches):
        return None
    first = dict(batches[0])
    group_id = str(first.get("group_id", ""))
    example_key = str(first.get("example_key", ""))
    if not group_id or not example_key:
        return None
    if any(str(batch.get("group_id", "")) != group_id or str(batch.get("example_key", "")) != example_key for batch in batches[1:]):
        return None
    return first


def available_preferred_metrics(context: Mapping[str, Any], metrics: list[str]) -> list[str]:
    preferred = [str(metric) for metric in context.get("preferred_metrics", [])]
    available = set(metrics)
    return [metric for metric in preferred if metric in available]
