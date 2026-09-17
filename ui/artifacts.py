"""Streamlit-independent MLflow artifact loading and preview helpers."""

from __future__ import annotations

import csv
import io
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import pandas as pd


DEFAULT_PREVIEW_BYTES = 1_000_000
DEFAULT_TEXT_CHARACTERS = 50_000
DEFAULT_TEXT_LINES = 1_000
DEFAULT_CSV_ROWS = 500
DEFAULT_CSV_COLUMNS = 100


class ArtifactKind(str, Enum):
    """Preview behavior for a supported artifact type."""

    IMAGE = "image"
    CSV = "csv"
    TEXT = "text"
    BINARY = "binary"


@dataclass(frozen=True)
class ArtifactMetadata:
    """Safe metadata for one MLflow artifact file."""

    path: str
    size: int | None = None

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).name


@dataclass
class ArtifactListResult:
    """Artifact metadata plus a non-fatal listing diagnostic."""

    run_id: str
    artifacts: list[ArtifactMetadata] = field(default_factory=list)
    error: str | None = None


@dataclass
class ArtifactDownload:
    """Bytes for the single artifact explicitly selected by the caller."""

    metadata: ArtifactMetadata
    data: bytes | None = None
    error: str | None = None

    @property
    def has_data(self) -> bool:
        return self.data is not None


@dataclass
class ArtifactPreview:
    """Bounded, display-ready content with no local filesystem metadata."""

    kind: ArtifactKind
    text: str | None = None
    dataframe: pd.DataFrame | None = None
    truncated: bool = False
    error: str | None = None


ClientFactory = Callable[[str | None], Any]


def _safe_artifact_path(value: Any) -> str:
    """Keep MLflow-logical paths while discarding accidental local paths."""

    raw = str(value or "").replace("\\", "/")
    parts = PurePosixPath(raw).parts
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw) or ".." in parts:
        return PurePosixPath(raw).name
    return "/".join(part for part in parts if part not in ("", "."))


def _artifact_size(item: Any) -> int | None:
    value = getattr(item, "file_size", None)
    if value is None and isinstance(item, dict):
        value = item.get("file_size")
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    return size if size >= 0 else None


def _read_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _mlflow_client(tracking_uri: str | None) -> Any:
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def list_run_artifacts(
    mlflow_run_id: str | None,
    tracking_uri: str | None = None,
    *,
    client_factory: ClientFactory | None = None,
) -> ArtifactListResult:
    """Recursively list artifact files without downloading their contents."""

    run_id = str(mlflow_run_id or "").strip()
    if not run_id:
        return ArtifactListResult(
            run_id="",
            error="MLflow run ID was not recorded for this run.",
        )

    try:
        factory = client_factory or _mlflow_client
        client = factory(str(tracking_uri).strip() or None)
    except Exception:
        return ArtifactListResult(
            run_id=run_id,
            error="MLflow is unavailable; artifacts cannot be loaded.",
        )

    artifacts: dict[str, ArtifactMetadata] = {}
    pending: list[str] = [""]
    visited: set[str] = set()
    try:
        while pending:
            directory = pending.pop()
            if directory in visited:
                continue
            visited.add(directory)
            for item in client.list_artifacts(run_id, directory or None):
                path = _safe_artifact_path(_read_field(item, "path"))
                if not path:
                    continue
                if bool(_read_field(item, "is_dir", False)):
                    pending.append(path)
                else:
                    artifacts[path] = ArtifactMetadata(path, _artifact_size(item))
    except Exception:
        return ArtifactListResult(
            run_id=run_id,
            artifacts=sorted(artifacts.values(), key=lambda item: item.path.casefold()),
            error="MLflow artifacts are temporarily unavailable.",
        )

    return ArtifactListResult(
        run_id=run_id,
        artifacts=sorted(artifacts.values(), key=lambda item: item.path.casefold()),
    )


def download_artifact(
    mlflow_run_id: str | None,
    tracking_uri: str | None,
    artifact: ArtifactMetadata,
    *,
    client_factory: ClientFactory | None = None,
) -> ArtifactDownload:
    """Download only ``artifact`` and return its bytes, never its local path."""

    run_id = str(mlflow_run_id or "").strip()
    if not run_id:
        return ArtifactDownload(
            artifact,
            error="MLflow run ID was not recorded for this run.",
        )

    try:
        factory = client_factory or _mlflow_client
        client = factory(str(tracking_uri).strip() or None)
        with tempfile.TemporaryDirectory(prefix="fedxplore-artifact-") as directory:
            local_path = client.download_artifacts(
                run_id,
                artifact.path,
                dst_path=directory,
            )
            data = Path(local_path).read_bytes()
    except Exception:
        return ArtifactDownload(
            artifact,
            error="The selected artifact could not be downloaded.",
        )
    return ArtifactDownload(artifact, data=data)


def classify_artifact(path: str) -> ArtifactKind:
    """Classify an artifact by its logical filename extension."""

    suffix = PurePosixPath(path).suffix.casefold()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return ArtifactKind.IMAGE
    if suffix == ".csv":
        return ArtifactKind.CSV
    if suffix in {".md", ".markdown", ".yaml", ".yml", ".json", ".txt"}:
        return ArtifactKind.TEXT
    return ArtifactKind.BINARY


def _bounded_text(
    data: bytes,
    *,
    max_bytes: int,
    max_characters: int,
    max_lines: int,
) -> tuple[str, bool]:
    bounded = data[:max_bytes]
    text = bounded.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    text = "".join(lines[:max_lines])[:max_characters]
    truncated = (
        len(data) > max_bytes
        or len(lines) > max_lines
        or len("".join(lines[:max_lines])) > max_characters
    )
    return text, truncated


def _bounded_csv(
    data: bytes,
    *,
    max_bytes: int,
    max_rows: int,
    max_columns: int,
) -> tuple[pd.DataFrame, bool]:
    bounded = data[:max_bytes]
    text = bounded.decode("utf-8-sig", errors="replace")
    rows = csv.reader(io.StringIO(text))
    sampled = []
    extra_row = False
    column_truncated = False
    for index, row in enumerate(rows):
        if index > max_rows:
            extra_row = True
            break
        column_truncated = column_truncated or len(row) > max_columns
        sampled.append(row[:max_columns])

    if not sampled:
        return pd.DataFrame(), len(data) > max_bytes
    width = min(max(len(row) for row in sampled), max_columns)
    header = sampled[0] + [""] * (width - len(sampled[0]))
    body = [row + [""] * (width - len(row)) for row in sampled[1 : max_rows + 1]]
    frame = pd.DataFrame(body, columns=header)
    return frame, len(data) > max_bytes or extra_row or column_truncated


def build_artifact_preview(
    download: ArtifactDownload,
    *,
    max_bytes: int = DEFAULT_PREVIEW_BYTES,
    max_text_characters: int = DEFAULT_TEXT_CHARACTERS,
    max_text_lines: int = DEFAULT_TEXT_LINES,
    max_csv_rows: int = DEFAULT_CSV_ROWS,
    max_csv_columns: int = DEFAULT_CSV_COLUMNS,
) -> ArtifactPreview:
    """Build a bounded preview from an explicitly downloaded artifact."""

    kind = classify_artifact(download.metadata.path)
    if download.error or download.data is None:
        return ArtifactPreview(kind, error=download.error or "Artifact data is unavailable.")

    if kind is ArtifactKind.IMAGE:
        if len(download.data) > max_bytes:
            return ArtifactPreview(
                kind,
                truncated=True,
                error="This image is too large to preview.",
            )
        return ArtifactPreview(kind)

    if kind is ArtifactKind.CSV:
        try:
            frame, truncated = _bounded_csv(
                download.data,
                max_bytes=max_bytes,
                max_rows=max_csv_rows,
                max_columns=max_csv_columns,
            )
        except (csv.Error, ValueError):
            return ArtifactPreview(kind, error="This CSV artifact could not be previewed.")
        return ArtifactPreview(kind, dataframe=frame, truncated=truncated)

    if kind is ArtifactKind.TEXT:
        text, truncated = _bounded_text(
            download.data,
            max_bytes=max_bytes,
            max_characters=max_text_characters,
            max_lines=max_text_lines,
        )
        return ArtifactPreview(kind, text=text, truncated=truncated)

    return ArtifactPreview(kind)
