# FedXplore Local UI

Local Streamlit dashboard for launching and monitoring `FedXplore` experiments from the same repository clone.

The UI does not replace the CLI. It builds the same Hydra override command for `src/train.py`, shows that command before launch, starts the process through `subprocess.Popen(...)`, and keeps run metadata under `outputs/ui/runs/`.

## Install

From the repository root:

```bash
pip install -e .
pip install -r ui/requirements-ui.txt
```

## Run

From the repository root:

```bash
streamlit run ui/run_ui.py
```

The UI starts in a light theme by default.

## Main flow

### Dashboard

The default page contains:

- summary cards for total runs / running / stopping
- one run table with name, method, dataset, created time, duration, status, clients, and rounds
- a hover-revealed selection checkbox for each run (the layout does not shift)
- `Create Run` action

Every row can be opened as a dedicated run page.

Select two or more runs to enable `Compare selected` in the page header. The comparison URL keeps
the selected run IDs, so a comparison can be bookmarked or shared. A single
run page also has a `Compare` action that opens the same view with only that
run preselected. The comparison page uses compact selected-run cards and an
`Add a run` picker; returning to Dashboard clears the temporary comparison
selection.

### Create Run

The create page is split into sections:

1. `Template`
2. `Experiment`
3. `Data & Clients`
4. `FL Method`
5. `Training`
6. `Attacks & Robustness`
7. `Tracking & Evaluation`
8. `Runtime & Resources`
9. `Review & Launch`

Navigation between steps is done through the clickable step selector and `Back` / `Next` buttons. The desktop layout keeps a compact Experiment Summary visible beside the current form; it reflects only high-signal research settings, not raw config paths.

The form is seeded from the repo Hydra defaults, so clicking through the default selections keeps you close to the baseline `python src/train.py` behavior.

The Review & Launch step shows the exact shell command that can be copied and started manually. Less frequently changed component parameters are kept under consistent Advanced settings expanders.

The `Technical` step provides a CPU / CUDA switch and GPU `device_ids` selection based on the GPUs visible via `nvidia-smi`.

Override priority is: template overrides, then structured form values, then manual raw overrides.

### Examples

`Examples` is a separate curated workflow, not another configuration form. It
launches one predefined suite of ordinary UI runs through the normal registry,
then opens Compare immediately with those runs selected. The initial catalog
contains the five-run *Client Selection × Byzantine Robustness* suite and the
six-run *Personalization vs Generalization* suite. Their scientific overrides
are stored in `ui/examples.yaml` and mirror the canonical toy shell scripts.
The suites run directly on CPU with local MLflow tracking; choosing a card
starts the suite immediately.

Each child stores durable `example_batch` metadata in `spec.yaml`, so Dashboard
and Compare can recover its method/dataset and curated context after refresh.
Examples always use MLflow and set per-child thread limits without modifying
the Streamlit server environment.

### Run page

Each run opens on its own page with:

- header with run name, status, and control buttons
- `Analytics` tab (the default)
- `Parameters` tab with structured subtabs
- `Git` tab
- `Logs`, `Journal`, `Files`, and `Overview` tabs

Available controls:

- `Stop`
- `Clone`
- `Re-run`
- `Compare`
- `Create Run` (primary action)
- `MLflow` link when available

`Clone` restores the same UI state into the create page and does not auto-start a new process.

`Re-run` immediately starts a new experiment from the source run's saved Hydra
overrides. It creates a fresh run ID/name and records `rerun_of` in its saved
metadata. It uses the currently checked-out FedXplore source code; it does not
check out or restore the source run's historical Git commit.

### Analytics and comparison

For MLflow runs, `Analytics` discovers metric names dynamically and reads their
full MLflow metric histories. It plots selected metrics in compact Plotly chart
cards against MLflow step;
when steps are missing or uninformative it uses timestamps. Completed runs also
show a final-metrics table, where “final” means the latest recorded history
point. Analytics refreshes automatically once per second while a run is
logging, including its status, MLflow ID, and newly available charts.

The comparison view overlays selected metrics for all selected runs in the same
chart cards, presents their latest metric values side by side, and compares
saved Hydra overrides. Only differing configuration values are shown by
default; enable `Show unchanged` to inspect the full saved configuration. Metrics missing
from a run display as `N/A`.

MLflow is optional. Runs without an MLflow logger/ID, runs that are still
starting, unavailable tracking stores, and legacy runs without MLflow metadata
remain inspectable and show an explanatory empty state instead of failing.

## Logs and run files

The main training log is written to:

```text
outputs/<run_name>__<run_id>.txt
```

The unique run-ID suffix prevents a later launch with the same display name
from replacing an older run's log. Legacy runs can still point to the older
`outputs/<run_name>.txt` format.

The run registry for the UI is stored in:

```text
outputs/ui/runs/<timestamp>_<sanitized_run_name>/
```

Inside that directory the UI keeps:

- `spec.yaml`
- `command.sh`
- `status.json`
- `pid.txt`
- `stdout.txt`
- `stderr.txt`
- `events.jsonl`
- `mlflow_url.txt` when present
- `provenance.json` for newly launched runs
- `git_diff.patch`, `git_diff_staged.patch`, and `git_diff_unstaged.patch`
  when provenance is captured

`stdout.txt` and `stderr.txt` point to the same primary log file.

### Git provenance

Every newly launched run captures Git provenance before its training process is
started. The snapshot records repository root, branch, full and short commit,
detached-HEAD state, `git describe`, origin URL, dirty state, porcelain status,
and modified/staged/unstaged/untracked file lists. Tracked changes are stored
as separate patch files; untracked file contents are intentionally not copied.

The `Git` tab displays only this saved snapshot. If a run was created
before provenance support, it says `Provenance was not captured for this run.`
and never substitutes the current repository state. Large patch previews are
truncated in that tab; full patches can be downloaded from `Files`.

## MLflow

If `logger=mlflow` is selected, the UI can:

- switch between remote tracking and a local file-backed store at `outputs/mlruns`
- prefill the remote tracking URI from the repo config or environment when available
- store an MLflow run link with the run when the logger reports it
- start a local `mlflow ui` automatically on the first `MLflow` click for a given store
- reuse that local MLflow UI process for the same store on later clicks
- disable proxy environment variables for the child process automatically
- add MLflow-related hosts to `NO_PROXY`

This is useful when the local shell session is behind a proxy but the remote MLflow server should be reached directly.

## GPU view

The `Technical` section contains a live GPU panel based on:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

It refreshes inside the create page and is only meant to help choose devices before launch.

## Templates

Templates live in `ui/templates/*.yaml`.
Selecting a template in the UI applies it immediately.

Each template can define:

```yaml
name: Friendly name
description: Optional text
form:
  run_name: example_run
  federated_method: fedavg
overrides:
  - model=resnet18
  - model_trainer=image
```

The UI applies `form` values to the visible fields and appends `overrides` into the raw override area.

## Repo-specific notes

This UI follows the current `FedXplore` config layout:

- `distribution.alpha` is used instead of `dataset.alpha`
- dataset selection is done through `dataset@train_dataset` / `dataset@test_dataset`
- manager batch size is configured through `manager.batch_generator.batch_size`
- method / logger / selector / optimizer / loss parameters are loaded from the corresponding Hydra config group files when available

## Known limitations

- no sweeps or multirun management yet
- no shared multi-user server
- no authentication
- no GPU locking
- no full Hydra schema auto-generation
- native analytics currently use MLflow scalar metric histories; non-MLflow
  logs are not parsed into charts
- Re-run intentionally uses the current checkout rather than automatically
  restoring a historical Git revision
