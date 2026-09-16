"""Small, MLflow-backed analytics primitives used by the Streamlit UI.

The module intentionally has no Streamlit dependency.  This keeps the metric
loading behaviour shared by the single-run and comparison views, and makes the
normalisation rules straightforward to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class MetricPoint:
    """One scalar metric observation recorded by MLflow."""

    metric: str
    value: float
    step: float | None
    timestamp: int | None
    run_id: str = ""


@dataclass(frozen=True)
class FinalMetric:
    """The last recorded value of one metric."""

    metric: str
    value: float
    step: float | None
    timestamp: int | None


@dataclass
class MetricLoadResult:
    """Metric histories plus a non-fatal diagnostic when loading was partial."""

    run_id: str
    points: list[MetricPoint] = field(default_factory=list)
    metric_names: list[str] = field(default_factory=list)
    error: str | None = None
    metric_errors: dict[str, str] = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return bool(self.points)


def _read_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _finite_float(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _integer_timestamp(value: Any) -> int | None:
    normalized = _finite_float(value)
    if normalized is None:
        return None
    return int(normalized)


def normalize_metric_history(
    metric: str,
    history: Iterable[Any],
    *,
    run_id: str = "",
) -> list[MetricPoint]:
    """Normalize MLflow metric entities (or equivalent mappings) to points.

    Invalid scalar values are ignored.  MLflow's timestamp is expressed in
    milliseconds; it remains in that representation until a view asks for a
    plotting dataframe.
    """

    points: list[MetricPoint] = []
    for item in history:
        value = _finite_float(_read_value(item, "value"))
        if value is None:
            continue
        points.append(
            MetricPoint(
                metric=str(_read_value(item, "key", metric) or metric),
                value=value,
                step=_finite_float(_read_value(item, "step")),
                timestamp=_integer_timestamp(_read_value(item, "timestamp")),
                run_id=run_id,
            )
        )
    return points


def load_metric_histories(
    mlflow_run_id: str | None,
    tracking_uri: str | None = None,
) -> MetricLoadResult:
    """Read all metric histories for an MLflow run without leaking failures.

    MLflow is imported lazily so opening a legacy non-MLflow run does not make
    the whole UI dependent on the package.  A tracking URI may be local or
    remote; the configured URI is passed untouched to ``MlflowClient``.
    """

    run_id = str(mlflow_run_id or "").strip()
    if not run_id:
        return MetricLoadResult(
            run_id="",
            error="MLflow run ID was not recorded for this run.",
        )

    try:
        from mlflow.tracking import MlflowClient
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        return MetricLoadResult(
            run_id=run_id,
            error=f"MLflow is unavailable: {exc}",
        )

    try:
        client = MlflowClient(tracking_uri=str(tracking_uri).strip() or None)
        mlflow_run = client.get_run(run_id)
    except Exception as exc:
        return MetricLoadResult(
            run_id=run_id,
            error=f"Could not load MLflow metrics: {exc}",
        )

    metrics = getattr(getattr(mlflow_run, "data", None), "metrics", {}) or {}
    metric_names = sorted(str(name) for name in metrics)
    result = MetricLoadResult(run_id=run_id, metric_names=metric_names)
    for metric in metric_names:
        try:
            result.points.extend(
                normalize_metric_history(
                    metric,
                    client.get_metric_history(run_id, metric),
                    run_id=run_id,
                )
            )
        except Exception as exc:
            result.metric_errors[metric] = str(exc)

    if metric_names and not result.points and result.metric_errors:
        result.error = "MLflow returned metric names, but their histories could not be read."
    return result


def metric_names(points: Iterable[MetricPoint]) -> list[str]:
    """Return metric names in a stable, human-friendly order."""

    return sorted({point.metric for point in points})


def uses_step_axis(points: Sequence[MetricPoint]) -> bool:
    """Whether the full series has usable MLflow training steps.

    A mixed series falls back to timestamps rather than silently plotting some
    points against rounds and others against time.
    """

    if not points or any(point.step is None for point in points):
        return False
    # A repeated default step (commonly zero) carries no ordering information
    # for a multi-point curve, so timestamp is the more honest axis.
    return len(points) == 1 or len({point.step for point in points}) > 1


def metric_points_frame(
    points: Iterable[MetricPoint],
    *,
    run_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return chart-ready points with either a step or timestamp X axis.

    For a comparison, the decision is made per metric across all selected
    runs.  If one run lacks usable steps, every line for that metric uses time
    so an Altair chart never receives mixed X-axis types.
    """

    grouped: dict[tuple[str, str], list[MetricPoint]] = {}
    for point in points:
        grouped.setdefault((point.run_id, point.metric), []).append(point)

    rows: list[dict[str, Any]] = []
    labels = run_labels or {}
    points_by_metric: dict[str, list[MetricPoint]] = {}
    for point in points:
        points_by_metric.setdefault(point.metric, []).append(point)
    axis_by_metric = {
        metric: uses_step_axis(series)
        for metric, series in points_by_metric.items()
    }

    for (run_id, metric), series in grouped.items():
        ordered = sorted(
            series,
            key=lambda point: (
                point.step is None if axis_by_metric[metric] else point.timestamp is None,
                (
                    point.step
                    if axis_by_metric[metric] and point.step is not None
                    else point.timestamp if point.timestamp is not None else float("inf")
                ),
                point.timestamp if point.timestamp is not None else -1,
            ),
        )
        use_step = axis_by_metric[metric]
        for point in ordered:
            timestamp_value = (
                datetime.fromtimestamp(point.timestamp / 1000, tz=timezone.utc)
                if point.timestamp is not None
                else None
            )
            rows.append(
                {
                    "run_id": run_id,
                    "run_label": labels.get(run_id, run_id or "Run"),
                    "metric": metric,
                    "value": point.value,
                    "step": point.step,
                    "timestamp": timestamp_value,
                    "x": point.step if use_step else timestamp_value,
                    "x_axis": "Step" if use_step else "Timestamp",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "run_label",
            "metric",
            "value",
            "step",
            "timestamp",
            "x",
            "x_axis",
        ],
    )


def load_final_metrics(points: Iterable[MetricPoint]) -> list[FinalMetric]:
    """Find the latest point for each metric by step, then timestamp.

    When MLflow did not record usable steps, timestamps provide a deterministic
    fallback.  This deliberately does not infer whether a metric should be
    minimized or maximized: "final" means last recorded value.
    """

    grouped: dict[str, list[MetricPoint]] = {}
    for point in points:
        grouped.setdefault(point.metric, []).append(point)

    final_metrics: list[FinalMetric] = []
    for metric, series in grouped.items():
        if uses_step_axis(series):
            latest = max(
                series,
                key=lambda point: (
                    point.step if point.step is not None else float("-inf"),
                    point.timestamp if point.timestamp is not None else -1,
                ),
            )
        else:
            latest = max(
                series,
                key=lambda point: point.timestamp if point.timestamp is not None else -1,
            )
        final_metrics.append(
            FinalMetric(
                metric=metric,
                value=latest.value,
                step=latest.step,
                timestamp=latest.timestamp,
            )
        )
    return sorted(final_metrics, key=lambda item: item.metric)
