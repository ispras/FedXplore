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
- `Create Run` action

Every row can be opened as a dedicated run page.

### Create Run

The create page is split into sections:

1. `Template`
2. `Run`
3. `Setup`
4. `Method`
5. `Logging`
6. `Training & Other`
7. `Attacks`
   This step exposes `clients_attack_types` and the base attack schedule fields from Hydra.
8. `Technical`
9. `Launch`

Navigation between steps is done through the step selector and `Back` / `Next` buttons.

The form is seeded from the repo Hydra defaults, so clicking through the default selections keeps you close to the baseline `python src/train.py` behavior.

The create page also shows the exact shell command that can be copied and started manually.

The `Technical` step provides a CPU / CUDA switch and GPU `device_ids` selection based on the GPUs visible via `nvidia-smi`.

Override priority is: template overrides, then structured form values, then manual raw overrides.

### Run page

Each run opens on its own page with:

- header with run name, status, and control buttons
- `Logs` tab
- `Parameters` tab with structured subtabs
- `Journal` tab
- `Files` tab
- `Overview` tab

Available controls:

- `Stop`
- `Clone`
- `Dashboard`
- `MLflow` link when available

`Clone` restores the same UI state into the create page and does not auto-start a new process.

## Logs and run files

The main training log is written to:

```text
outputs/<run_name>.txt
```

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

`stdout.txt` and `stderr.txt` point to the same primary log file.

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
- no advanced run comparison yet
