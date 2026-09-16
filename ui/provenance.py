"""Git provenance snapshots for UI-launched FedXplore runs.

The functions in this module intentionally do not depend on Streamlit or the
launcher.  A snapshot is collected at launch time and persisted beside the
run, so later views never have to infer historical Git state from the current
working tree.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_FILE_NAME = "provenance.json"
COMBINED_DIFF_FILE_NAME = "git_diff.patch"
STAGED_DIFF_FILE_NAME = "git_diff_staged.patch"
UNSTAGED_DIFF_FILE_NAME = "git_diff_unstaged.patch"
DEFAULT_GIT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class _GitResult:
    """The result of a Git command without allowing subprocess failures through."""

    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    error: str | None = None
    executable_missing: bool = False

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.returncode == 0


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_git(
    repo_root: Path, args: Sequence[str], *, timeout_seconds: float
) -> _GitResult:
    """Run Git with a fixed argv and turn operational failures into data."""

    command = ["git", "-C", str(repo_root), *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return _GitResult(
            error="Git executable is not available.", executable_missing=True
        )
    except subprocess.TimeoutExpired as exc:
        return _GitResult(
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            error=f"Git command timed out after {timeout_seconds:g} seconds.",
        )
    except OSError as exc:
        return _GitResult(error=f"Could not execute Git: {exc}")

    return _GitResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _result_error(result: _GitResult, fallback: str) -> str:
    if result.error:
        return result.error
    details = result.stderr.strip() or result.stdout.strip()
    if details:
        return f"{fallback}: {details}"
    if result.returncode is not None:
        return f"{fallback} (exit code {result.returncode})."
    return fallback


def _optional_git_value(
    repo_root: Path, args: Sequence[str], *, timeout_seconds: float
) -> str | None:
    result = _run_git(repo_root, args, timeout_seconds=timeout_seconds)
    if not result.succeeded:
        return None
    value = result.stdout.strip()
    return value or None


def _parse_status_porcelain(raw_status: str) -> dict[str, list[str]]:
    """Normalize ``git status --porcelain=v1 -z`` output for JSON/UI use."""

    status_porcelain: list[str] = []
    modified_files: list[str] = []
    staged_files: list[str] = []
    unstaged_files: list[str] = []
    untracked_files: list[str] = []

    entries = raw_status.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry or len(entry) < 3:
            continue

        status = entry[:2]
        path = entry[3:]
        original_path: str | None = None
        if ("R" in status or "C" in status) and index < len(entries):
            original_path = entries[index]
            index += 1

        display_path = path
        if original_path:
            display_path = f"{original_path} -> {path}"
        status_porcelain.append(f"{status} {display_path}")

        if status == "??":
            untracked_files.append(path)
            continue
        if status == "!!":
            continue

        modified_files.append(display_path)
        if status[0] != " ":
            staged_files.append(display_path)
        if status[1] != " ":
            unstaged_files.append(display_path)

    return {
        "status_porcelain": status_porcelain,
        "modified_files": sorted(set(modified_files)),
        "staged_files": sorted(set(staged_files)),
        "unstaged_files": sorted(set(unstaged_files)),
        "untracked_files": sorted(set(untracked_files)),
    }


def _empty_git_data(
    repo_root: Path, *, available: bool, repository: bool, error: str | None
) -> dict[str, Any]:
    return {
        "available": available,
        "repository": repository,
        "repo_root": str(repo_root),
        "branch": None,
        "commit": None,
        "short_commit": None,
        "detached_head": False,
        "describe": None,
        "remote_origin": None,
        "dirty": None,
        "status_porcelain": [],
        "modified_files": [],
        "staged_files": [],
        "unstaged_files": [],
        "untracked_files": [],
        "error": error,
    }


def _collect_provenance_snapshot(
    repo_root: Path | str, *, timeout_seconds: float
) -> tuple[dict[str, Any], dict[str, str]]:
    requested_root = Path(repo_root).expanduser().resolve()
    patch_files = {
        "combined": COMBINED_DIFF_FILE_NAME,
        "staged": STAGED_DIFF_FILE_NAME,
        "unstaged": UNSTAGED_DIFF_FILE_NAME,
    }
    empty_patches = {name: "" for name in patch_files}

    root_result = _run_git(
        requested_root,
        ["rev-parse", "--show-toplevel"],
        timeout_seconds=timeout_seconds,
    )
    if not root_result.succeeded:
        git_data = _empty_git_data(
            requested_root,
            available=not root_result.executable_missing,
            repository=False,
            error=_result_error(root_result, "Could not identify a Git repository"),
        )
        return (
            {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "captured_at": _iso_now(),
                "git": git_data,
                "patch_files": patch_files,
            },
            empty_patches,
        )

    actual_root = Path(root_result.stdout.strip()).resolve()
    status_result = _run_git(
        actual_root,
        ["status", "--porcelain=v1", "-z"],
        timeout_seconds=timeout_seconds,
    )
    status_data = (
        _parse_status_porcelain(status_result.stdout)
        if status_result.succeeded
        else {
            "status_porcelain": [],
            "modified_files": [],
            "staged_files": [],
            "unstaged_files": [],
            "untracked_files": [],
        }
    )

    commit = _optional_git_value(
        actual_root,
        ["rev-parse", "HEAD"],
        timeout_seconds=timeout_seconds,
    )
    short_commit = _optional_git_value(
        actual_root,
        ["rev-parse", "--short", "HEAD"],
        timeout_seconds=timeout_seconds,
    )
    branch = _optional_git_value(
        actual_root,
        ["branch", "--show-current"],
        timeout_seconds=timeout_seconds,
    )
    describe = _optional_git_value(
        actual_root,
        ["describe", "--always", "--dirty", "--tags"],
        timeout_seconds=timeout_seconds,
    )
    remote_origin = _optional_git_value(
        actual_root,
        ["remote", "get-url", "origin"],
        timeout_seconds=timeout_seconds,
    )

    combined_diff = ""
    if commit:
        combined_result = _run_git(
            actual_root,
            ["diff", "--no-ext-diff", "--no-textconv", "HEAD"],
            timeout_seconds=timeout_seconds,
        )
        if combined_result.succeeded:
            combined_diff = combined_result.stdout

    staged_result = _run_git(
        actual_root,
        ["diff", "--no-ext-diff", "--no-textconv", "--cached"],
        timeout_seconds=timeout_seconds,
    )
    unstaged_result = _run_git(
        actual_root,
        ["diff", "--no-ext-diff", "--no-textconv"],
        timeout_seconds=timeout_seconds,
    )
    patches = {
        "combined": combined_diff,
        "staged": staged_result.stdout if staged_result.succeeded else "",
        "unstaged": unstaged_result.stdout if unstaged_result.succeeded else "",
    }

    git_data: dict[str, Any] = {
        "available": True,
        "repository": True,
        "repo_root": str(actual_root),
        "branch": branch,
        "commit": commit,
        "short_commit": short_commit,
        "detached_head": bool(commit and not branch),
        "describe": describe,
        "remote_origin": remote_origin,
        "dirty": bool(status_data["status_porcelain"])
        if status_result.succeeded
        else None,
        **status_data,
        "error": None,
    }
    if not status_result.succeeded:
        git_data["error"] = _result_error(
            status_result, "Could not read Git working-tree status"
        )

    return (
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "captured_at": _iso_now(),
            "git": git_data,
            "patch_files": patch_files,
        },
        patches,
    )


def collect_provenance(
    repo_root: Path | str, *, timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Collect a serializable Git snapshot without writing any run files.

    Git/network/storage failures are represented in the returned ``git``
    payload instead of being raised.  Diffs are intentionally omitted from the
    returned JSON-ready payload; :func:`write_provenance` persists them as
    separate patch files.
    """

    provenance, _ = _collect_provenance_snapshot(
        repo_root, timeout_seconds=timeout_seconds
    )
    return provenance


def write_provenance(
    run_dir: Path | str,
    repo_root: Path | str,
    *,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Capture provenance and persist JSON plus combined/staged/unstaged diffs."""

    provenance, patches = _collect_provenance_snapshot(
        repo_root, timeout_seconds=timeout_seconds
    )
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    patch_files = provenance["patch_files"]
    for patch_kind, file_name in patch_files.items():
        (target_dir / file_name).write_text(patches[patch_kind], encoding="utf-8")

    (target_dir / PROVENANCE_FILE_NAME).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def load_provenance(run_dir: Path | str) -> dict[str, Any] | None:
    """Load a saved snapshot, returning ``None`` for legacy or unreadable runs."""

    path = Path(run_dir) / PROVENANCE_FILE_NAME
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None
