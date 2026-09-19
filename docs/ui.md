# UI technical reference

This document describes the runtime and persistence behavior of the FedXplore
UI. For installation and the interactive workflow, start with the
[main README](../README.md#create-and-inspect-a-run).

## Runtime model

Start the local Streamlit application from the repository root:

```bash
python -m streamlit run ui/run_ui.py
```

The UI resolves the same Hydra configuration used by `src/train.py`, displays
the resulting command before launch, and starts training through
`subprocess.Popen(...)`. UI runs therefore use the regular training path rather
than a separate execution backend.

Override priority is:

1. template overrides
2. values selected in the structured form
3. manual raw overrides

## Run state and logs

The UI registry stores each run under:

```text
outputs/ui/runs/<timestamp>_<sanitized_run_name>/
```

Depending on the run state, the directory contains:

- `spec.yaml` with the saved launch specification
- `command.sh` with the resolved command
- `status.json` and `pid.txt` with process state
- `events.jsonl` with lifecycle events
- `mlflow_url.txt` when an MLflow run is available
- `provenance.json` and Git patch files for captured source state
- `stdout.txt` and `stderr.txt` pointing to the primary training log

The primary log uses a unique run ID to avoid replacing an earlier run with the
same display name:

```text
outputs/<run_name>__<run_id>.txt
```

## Run lifecycle

**Clone** restores a saved run into the Create Run form without starting it.

**Re-run** immediately creates a new run from the saved Hydra overrides and
records `rerun_of` in its metadata. It executes the currently checked-out
FedXplore source; it does not restore the historical Git revision of the source
run.

**Compare** opens selected runs in a shared metric and configuration view. The
comparison shows changed overrides by default and can include unchanged values
when needed.

## Analytics and artifacts

For MLflow runs, the UI reads metric histories and artifacts through the MLflow
client API. This works with local and remote tracking stores without requiring
the separate MLflow web interface.

Metric charts use recorded steps when available and timestamps otherwise.
Completed runs show the latest recorded value for each metric. While training
is active, status and metric histories refresh automatically.

The artifact browser lists files before downloading them and provides bounded
previews for images, CSV, Markdown, YAML, JSON, and plain text. Runs without
MLflow metadata remain available for logs, configuration, files, and lifecycle
actions.

## Git provenance

Before a newly launched process starts, the UI records the current branch,
commit, origin, dirty state, and changed file lists. Tracked changes are stored
as staged and unstaged patches. Untracked file names are recorded, but their
contents are not copied.

The Git view always displays the snapshot saved with the run. Legacy runs
without a snapshot do not substitute the current repository state.

## Examples and templates

Curated example suites are defined in [`ui/examples.yaml`](../ui/examples.yaml).
Each suite launches ordinary registered runs and stores `example_batch`
metadata so its comparison can be recovered after a refresh.

Templates live in `ui/templates/*.yaml`. A template can provide initial form
values and Hydra overrides; subsequent form edits and raw overrides take
precedence.

## Runtime devices

The Create Run page discovers visible GPUs with `nvidia-smi` and exposes their
IDs in the runtime controls. This is an informational view for choosing a
device; the UI does not reserve or lock GPUs.

## Current limitations

- no sweeps or multirun management
- no shared multi-user server or authentication
- no GPU locking
- no complete Hydra schema generation
- native charts require MLflow scalar metric histories
- Re-run uses the current checkout instead of restoring a historical revision
