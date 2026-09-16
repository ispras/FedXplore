"""Offline curated documentation for selectable research components."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import yaml


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def readable_name(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"[_-]+", " ", text).title() or "Not configured"


def load_research_catalog(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load static metadata, returning an empty catalog for a missing file."""

    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections = data.get("sections", data)
    if not isinstance(sections, dict):
        raise ValueError("research catalog must contain a mapping of sections")
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    for section, entries in sections.items():
        if not isinstance(entries, dict):
            continue
        catalog[str(section)] = {
            str(name): dict(metadata)
            for name, metadata in entries.items()
            if isinstance(metadata, dict)
        }
    return catalog


def metadata_for(
    catalog: Mapping[str, Mapping[str, Mapping[str, Any]]],
    section: str,
    option: str,
) -> dict[str, Any]:
    """Resolve an option through its canonical key or aliases with a fallback."""

    wanted = _key(option)
    for name, raw_metadata in catalog.get(section, {}).items():
        metadata = dict(raw_metadata)
        aliases = [name, *metadata.get("aliases", [])]
        if any(_key(alias) == wanted for alias in aliases):
            metadata.setdefault("display_name", readable_name(option))
            metadata.setdefault("aliases", [])
            metadata["known"] = True
            return metadata
    return {
        "display_name": readable_name(option),
        "description": "No curated description is available for this component yet.",
        "tags": [],
        "reference": None,
        "known": False,
    }


def ordered_options(
    catalog: Mapping[str, Mapping[str, Mapping[str, Any]]],
    section: str,
    options: list[str],
) -> list[str]:
    """Keep curated components first while always retaining unknown local ones."""

    known: list[tuple[int, str]] = []
    unknown: list[str] = []
    seen_known: set[str] = set()
    for option in options:
        metadata = metadata_for(catalog, section, option)
        if metadata["known"]:
            canonical = _key(metadata.get("display_name"))
            if canonical in seen_known:
                continue
            seen_known.add(canonical)
            known.append((int(metadata.get("order", 10_000)), option))
        else:
            unknown.append(option)
    return [option for _, option in sorted(known)] + sorted(unknown, key=_key)
