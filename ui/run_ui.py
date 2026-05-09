from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

try:
    from .launcher import (
        DEFAULT_RUN_NAME,
        TemplateSpec,
        build_command,
        build_overrides,
        build_subprocess_env,
        discover_config_group_options,
        extract_main_default_selections,
        find_duplicate_override_keys,
        flatten_mapping,
        format_manual_shell_command,
        get_component_default_params,
        get_main_config,
        get_mlflow_defaults,
        get_repo_root,
        list_runs,
        load_templates,
        normalize_mlflow_ui_url,
        parse_iso_datetime,
        parse_raw_overrides,
        preview_stdout_path,
        query_gpus,
        read_run_events,
        read_spec,
        start_run,
        stop_run,
        tail_file,
        unflatten_mapping,
    )
except ImportError:
    from launcher import (
        DEFAULT_RUN_NAME,
        TemplateSpec,
        build_command,
        build_overrides,
        build_subprocess_env,
        discover_config_group_options,
        extract_main_default_selections,
        find_duplicate_override_keys,
        flatten_mapping,
        format_manual_shell_command,
        get_component_default_params,
        get_main_config,
        get_mlflow_defaults,
        get_repo_root,
        list_runs,
        load_templates,
        normalize_mlflow_ui_url,
        parse_iso_datetime,
        parse_raw_overrides,
        preview_stdout_path,
        query_gpus,
        read_run_events,
        read_spec,
        start_run,
        stop_run,
        tail_file,
        unflatten_mapping,
    )


VIEW_KEY = "ui_view"
VIEW_DASHBOARD = "dashboard"
VIEW_CREATE = "create_run"
VIEW_RUN = "run_detail"

SELECTED_RUN_KEY = "ui_selected_run_id"
LAST_LINES_KEY = "ui_last_log_lines"
TEMPLATE_PICKER_KEY = "ui_template_picker"
LOADED_TEMPLATE_KEY = "ui_loaded_template"
CREATE_STEP_KEY = "ui_create_step"
PENDING_TEMPLATE_KEY = "ui_pending_template_key"
PENDING_RESET_KEY = "ui_pending_form_reset"
PENDING_STEP_KEY = "ui_pending_target_step"
DEVICE_MODE_KEY = "ui_device_mode"
DEVICE_IDS_KEY = "ui_device_ids_selected"
ATTACK_TYPE_KEY = "ui_attack_type"
TEMPLATE_OVERRIDES_KEY = "ui_template_overrides_text"
DASHBOARD_NAME_FILTER_KEY = "ui_dashboard_filter_name"
DASHBOARD_METHOD_FILTER_KEY = "ui_dashboard_filter_method"
DASHBOARD_DATASET_FILTER_KEY = "ui_dashboard_filter_dataset"
DASHBOARD_STATUS_FILTER_KEY = "ui_dashboard_filter_status"

GENERAL_UI_KEYS = {
    VIEW_KEY,
    SELECTED_RUN_KEY,
    LAST_LINES_KEY,
    TEMPLATE_PICKER_KEY,
    LOADED_TEMPLATE_KEY,
    CREATE_STEP_KEY,
    PENDING_TEMPLATE_KEY,
    PENDING_RESET_KEY,
    PENDING_STEP_KEY,
    DASHBOARD_NAME_FILTER_KEY,
    DASHBOARD_METHOD_FILTER_KEY,
    DASHBOARD_DATASET_FILTER_KEY,
    DASHBOARD_STATUS_FILTER_KEY,
}

SELECTION_KEYS = {
    "train_dataset": "ui_train_dataset",
    "test_dataset": "ui_test_dataset",
    "trust_dataset": "ui_trust_dataset",
    "distribution": "ui_distribution",
    "model": "ui_model",
    "model_trainer": "ui_model_trainer",
    "federated_method": "ui_federated_method",
    "client_selector": "ui_client_selector",
    "preaggregator": "ui_preaggregator",
    "logger": "ui_logger",
    "optimizer": "ui_optimizer",
    "loss": "ui_loss",
    "manager": "ui_manager",
    "manager_batch_generator": "ui_manager_batch_generator",
}

COMPONENT_LABELS = {
    "distribution": "Distribution",
    "model": "Model",
    "model_trainer": "Model trainer",
    "federated_method": "Method",
    "client_selector": "Client selection",
    "preaggregator": "Preaggregator",
    "logger": "Logger",
    "optimizer": "Optimizer",
    "loss": "Loss",
    "manager": "Manager",
    "manager_batch_generator": "Batch generator",
}

SETUP_BASE_PATHS = [
    "training_params.batch_size",
    "training_params.num_workers",
    "federated_params.amount_of_clients",
    "federated_params.client_subset_size",
    "federated_params.communication_rounds",
    "federated_params.local_epochs",
    "federated_params.client_train_val_prop",
]
OTHER_BASE_PATHS = [
    "federated_params.print_client_metrics",
    "federated_params.server_saving_metrics",
    "federated_params.server_saving_agg",
]
PARAMETER_TAB_COMPONENTS = [
    ("Setup", ["distribution", "model", "model_trainer"]),
    ("Method", ["federated_method", "client_selector", "preaggregator"]),
    ("Logging", ["logger"]),
    ("Training", ["optimizer", "loss"]),
    ("Attacks", []),
    ("Technical", ["manager", "manager_batch_generator"]),
]
CREATE_STEPS = [
    "template",
    "run",
    "setup",
    "method",
    "logging",
    "training",
    "attacks",
    "technical",
    "launch",
]
CREATE_STEP_LABELS = {
    "template": "1. Template",
    "run": "2. Run",
    "setup": "3. Setup",
    "method": "4. Method",
    "logging": "5. Logging",
    "training": "6. Training",
    "attacks": "7. Attacks",
    "technical": "8. Technical",
    "launch": "9. Launch",
}


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
        return
    st.experimental_rerun()


def install_keyboard_guard() -> None:
    st.html(
        """
        <script>
        (function () {
          const doc = window.document;
          if (doc.__fedxploreKeyboardGuardInstalled) {
            return;
          }
          doc.__fedxploreKeyboardGuardInstalled = true;
          doc.addEventListener("keydown", function (event) {
            const key = String(event.key || "").toLowerCase();
            const target = event.target;
            const tag = target && target.tagName ? target.tagName.toLowerCase() : "";
            const editable = !!(
              target &&
              (target.isContentEditable ||
                tag === "input" ||
                tag === "textarea" ||
                tag === "select")
            );
            if (editable) {
              return;
            }
            if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey && key === "c") {
              event.preventDefault();
              event.stopPropagation();
              event.stopImmediatePropagation();
            }
          }, true);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def sync_query_params(view: str, run_id: str | None = None) -> None:
    if hasattr(st, "query_params"):
        st.query_params.clear()
        st.query_params["view"] = view
        if run_id:
            st.query_params["run_id"] = run_id
        return
    params: dict[str, str] = {"view": view}
    if run_id:
        params["run_id"] = run_id
    st.experimental_set_query_params(**params)


def restore_view_from_query_params() -> None:
    if hasattr(st, "query_params"):
        view = st.query_params.get("view", "")
        run_id = st.query_params.get("run_id", "")
    else:
        params = st.experimental_get_query_params()
        view = params.get("view", [""])[0]
        run_id = params.get("run_id", [""])[0]
    if view in {VIEW_DASHBOARD, VIEW_CREATE, VIEW_RUN}:
        st.session_state[VIEW_KEY] = view
    if run_id:
        st.session_state[SELECTED_RUN_KEY] = run_id


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            display: none;
        }
        div[data-testid="stDecoration"] {
            display: none;
        }
        .stApp {
            background: linear-gradient(180deg, #f6fbff 0%, #ffffff 24%);
            color: #173456;
        }
        .main .block-container {
            padding-top: 0.05rem !important;
            padding-bottom: 1rem;
            max-width: 1500px;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #d8e7fb;
        }
        .stButton > button {
            background: linear-gradient(135deg, #2d6df6 0%, #4a8aff 100%);
            color: white;
            border: 0;
            border-radius: 12px;
            font-weight: 600;
            min-height: 2.65rem;
            box-shadow: 0 10px 26px rgba(45, 109, 246, 0.14);
        }
        .stButton > button:hover {
            filter: brightness(1.02);
        }
        .fx-topline {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.35rem;
        }
        .fx-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #d7e6fb;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 26px rgba(25, 79, 173, 0.06);
        }
        .fx-card-label {
            color: #5b7ea7;
            font-size: 0.84rem;
            margin-bottom: 0.35rem;
        }
        .fx-card-value {
            color: #173456;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.05;
        }
        .fx-status {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.7rem;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .fx-status.running { background: #e9f2ff; color: #195ad7; }
        .fx-status.stopping { background: #fff5e8; color: #b46d00; }
        .fx-status.stopped,
        .fx-status.finished { background: #eef3f9; color: #5c6f87; }
        .fx-status.failed_to_start,
        .fx-status.missing_status,
        .fx-status.invalid_status,
        .fx-status.missing_pid { background: #ffe9ee; color: #b42338; }
        .fx-status.default { background: #eef3f9; color: #5c6f87; }
        .fx-table {
            border: 1px solid #d7e6fb;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(20, 77, 169, 0.06);
            padding: 0.35rem 0.45rem 0.45rem 0.45rem;
        }
        .fx-table-header {
            color: #4a6c95;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }
        .fx-divider {
            height: 1px;
            background: #edf3fb;
            margin: 0.25rem 0;
        }
        .fx-section {
            margin-top: 0.5rem;
            padding-top: 0.15rem;
        }
        .fx-param-card {
            border: 1px solid #dde8f8;
            border-radius: 16px;
            padding: 0.8rem 0.9rem 0.25rem 0.9rem;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(20, 77, 169, 0.04);
            margin-bottom: 0.7rem;
        }
        .fx-run-head {
            margin-bottom: 0.7rem;
        }
        .fx-kv {
            display: grid;
            grid-template-columns: 180px 1fr;
            gap: 0.5rem 1rem;
            align-items: start;
        }
        .fx-kv-label {
            color: #5d7ca1;
            font-weight: 600;
        }
        .fx-detail-hero {
            padding: 0.15rem 0 0.9rem 0;
            border-bottom: 1px solid #e5eef9;
            margin-bottom: 1rem;
        }
        .fx-detail-title {
            color: #173456;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }
        .fx-detail-subtitle {
            color: #6886aa;
            font-size: 0.95rem;
        }
        .fx-detail-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem 1.35rem;
            margin-top: 1rem;
        }
        .fx-detail-item {
            min-width: 0;
        }
        .fx-detail-label {
            color: #6f89a8;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }
        .fx-detail-value {
            color: #173456;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.35;
            word-break: break-word;
        }
        .fx-step-note {
            color: #6f89a8;
            margin-bottom: 0.55rem;
        }
        .fx-gpu-panel {
            border: 1px solid #d8e6fb;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            margin-bottom: 0.65rem;
        }
        .fx-gpu-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.55rem;
            font-weight: 600;
            color: #18385f;
        }
        .fx-gpu-bar {
            height: 10px;
            border-radius: 999px;
            background: #e5eefc;
            overflow: hidden;
            margin-bottom: 0.45rem;
        }
        .fx-gpu-bar > span {
            display: block;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2d6df6 0%, #62a0ff 100%);
        }
        button[data-baseweb="tab"] {
            min-height: 3.2rem;
            padding: 0.85rem 1.2rem;
            font-size: 1rem;
            font-weight: 700;
        }
        [data-testid="stRadio"] label {
            padding-top: 0.15rem;
            padding-bottom: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pick_options(discovered: list[str], fallback: list[str]) -> list[str]:
    return discovered or fallback


def get_option_sets(repo_root: Path) -> dict[str, list[str]]:
    return {
        "dataset": pick_options(
            discover_config_group_options(
                repo_root, "dataset", exclude={"federated_dataset"}
            ),
            ["cifar10", "cifar100", "ptbxl", "tiny_imagenet", "tis"],
        ),
        "distribution": pick_options(
            discover_config_group_options(repo_root, "distribution"),
            ["uniform", "dirichlet", "sharded"],
        ),
        "model": pick_options(
            discover_config_group_options(repo_root, "model"),
            ["resnet18"],
        ),
        "model_trainer": pick_options(
            discover_config_group_options(repo_root, "model_trainer"),
            ["image"],
        ),
        "federated_method": pick_options(
            discover_config_group_options(repo_root, "federated_method"),
            ["fedavg", "fedamp", "fltrust"],
        ),
        "client_selector": pick_options(
            discover_config_group_options(repo_root, "client_selector", exclude={"base"}),
            ["uniform", "pow", "fedcor"],
        ),
        "preaggregator": [""] + pick_options(
            discover_config_group_options(repo_root, "preaggregator"),
            ["bucketing", "fbm"],
        ),
        "logger": pick_options(
            discover_config_group_options(repo_root, "logger"),
            ["base", "mlflow"],
        ),
        "optimizer": pick_options(
            discover_config_group_options(repo_root, "optimizer"),
            ["adam", "sgd"],
        ),
        "loss": pick_options(
            discover_config_group_options(repo_root, "losses"),
            ["ce", "bce"],
        ),
        "manager": pick_options(
            discover_config_group_options(repo_root, "manager"),
            ["base_manager"],
        ),
        "manager_batch_generator": pick_options(
            discover_config_group_options(repo_root, "manager/batch_generator"),
            ["sequential"],
        ),
        "attack_type": ["no_attack"]
        + pick_options(
            discover_config_group_options(repo_root, "attacks"),
            ["label_flip", "sign_flip", "random_grad", "alie", "ipm"],
        ),
    }


def dump_complex_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        text = yaml.safe_dump(
            value,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        ).strip()
        return text or "null"
    return str(value)


def safe_key(raw: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in raw)


def base_widget_key(path: str) -> str:
    return f"ui_base__{safe_key(path)}"


def component_widget_key(component: str, option: str, path: str) -> str:
    option_key = safe_key(option or "none")
    return f"ui_comp__{component}__{option_key}__{safe_key(path)}"


def widget_seed_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    return dump_complex_value(value)


def seed_state_value(key: str, value: Any) -> None:
    st.session_state.setdefault(key, widget_seed_value(value))


def parse_widget_value(raw_value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return bool(raw_value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, str):
        return str(raw_value)
    return yaml.safe_load(str(raw_value))


def ensure_state_defaults(defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def keep_ui_state_alive() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("ui_"):
            st.session_state[key] = st.session_state[key]


def build_default_state(repo_root: Path, options: dict[str, list[str]]) -> dict[str, Any]:
    config = get_main_config(repo_root)
    default_selections = extract_main_default_selections(repo_root)
    mlflow_defaults = get_mlflow_defaults(repo_root)
    main_flat_defaults = flatten_mapping(
        {
            key: value
            for key, value in config.items()
            if key not in {"defaults", "single_run_dir"}
        }
    )

    defaults = {
        VIEW_KEY: VIEW_DASHBOARD,
        SELECTED_RUN_KEY: "",
        LAST_LINES_KEY: 200,
        TEMPLATE_PICKER_KEY: "",
        LOADED_TEMPLATE_KEY: "",
        CREATE_STEP_KEY: CREATE_STEPS[0],
        PENDING_TEMPLATE_KEY: "",
        PENDING_RESET_KEY: False,
        PENDING_STEP_KEY: "",
        "ui_run_name": DEFAULT_RUN_NAME,
        "ui_mlflow_ui_url": mlflow_defaults["ui_url"],
        "ui_raw_overrides": "",
        TEMPLATE_OVERRIDES_KEY: "",
        DEVICE_MODE_KEY: str(main_flat_defaults.get("training_params.device", "cuda")),
        DEVICE_IDS_KEY: list(main_flat_defaults.get("training_params.device_ids", [0])),
        ATTACK_TYPE_KEY: (
            main_flat_defaults.get("federated_params.clients_attack_types", "no_attack")[0]
            if isinstance(
                main_flat_defaults.get("federated_params.clients_attack_types", "no_attack"),
                list,
            )
            and main_flat_defaults.get("federated_params.clients_attack_types", "no_attack")
            else str(main_flat_defaults.get("federated_params.clients_attack_types", "no_attack"))
        ),
        DASHBOARD_NAME_FILTER_KEY: "",
        DASHBOARD_METHOD_FILTER_KEY: "All",
        DASHBOARD_DATASET_FILTER_KEY: "All",
        DASHBOARD_STATUS_FILTER_KEY: "All",
    }

    for selection_key, state_key in SELECTION_KEYS.items():
        fallback_options = options.get(selection_key, [])
        if selection_key in {"preaggregator", "trust_dataset"}:
            defaults[state_key] = default_selections.get(selection_key, "")
            continue
        defaults[state_key] = default_selections.get(selection_key) or (
            fallback_options[0] if fallback_options else ""
        )

    base_config = {
        key: value
        for key, value in config.items()
        if key not in {"defaults", "single_run_dir"}
    }
    for path, value in flatten_mapping(base_config).items():
        defaults[base_widget_key(path)] = widget_seed_value(value)

    return defaults

def navigate_to(view: str, *, run_id: str | None = None) -> None:
    st.session_state[VIEW_KEY] = view
    if run_id is not None:
        st.session_state[SELECTED_RUN_KEY] = run_id
    sync_query_params(view, run_id)


def reset_form_to_defaults(defaults: dict[str, Any]) -> None:
    keys_to_remove = [
        key
        for key in st.session_state
        if key.startswith("ui_") and key not in GENERAL_UI_KEYS
    ]
    for key in keys_to_remove:
        del st.session_state[key]
    for key, value in defaults.items():
        if key.startswith("ui_") and key not in GENERAL_UI_KEYS:
            st.session_state[key] = value
    st.session_state[LOADED_TEMPLATE_KEY] = ""


def apply_template_to_state(
    template: TemplateSpec,
    defaults: dict[str, Any],
) -> None:
    reset_form_to_defaults(defaults)
    field_map = {
        "run_name": "ui_run_name",
        "dataset": ("ui_train_dataset", "ui_test_dataset"),
        "trust_dataset": "ui_trust_dataset",
        "distribution": "ui_distribution",
        "federated_method": "ui_federated_method",
        "client_selector": "ui_client_selector",
        "logger": "ui_logger",
        "model": "ui_model",
        "model_trainer": "ui_model_trainer",
        "optimizer": "ui_optimizer",
        "loss": "ui_loss",
        "preaggregator": "ui_preaggregator",
        "attack_type": ATTACK_TYPE_KEY,
    }
    scalar_path_map = {
        "random_state": "random_state",
        "communication_rounds": "federated_params.communication_rounds",
        "amount_of_clients": "federated_params.amount_of_clients",
        "client_subset_size": "federated_params.client_subset_size",
        "training_batch_size": "training_params.batch_size",
        "manager_batch_size": ("manager_batch_generator", "batch_size"),
        "device_mode": "training_params.device",
        "device_ids": "training_params.device_ids",
        "print_client_metrics": "federated_params.print_client_metrics",
        "distribution_alpha": ("distribution", "alpha"),
        "attack_scheme": "federated_params.attack_scheme",
        "prop_attack_clients": "federated_params.prop_attack_clients",
        "prop_attack_rounds": "federated_params.prop_attack_rounds",
    }

    for field, value in template.form.items():
        if field in field_map:
            target = field_map[field]
            if isinstance(target, tuple):
                for state_key in target:
                    st.session_state[state_key] = value
            else:
                st.session_state[target] = value
            continue

        if field in scalar_path_map:
            mapped = scalar_path_map[field]
            if mapped == "training_params.device":
                st.session_state[DEVICE_MODE_KEY] = (
                    "cuda" if str(value).lower() in {"gpu", "cuda"} else "cpu"
                )
            elif mapped == "training_params.device_ids":
                st.session_state[DEVICE_IDS_KEY] = normalize_device_selection(value)
            elif isinstance(mapped, tuple):
                component_name, param_path = mapped
                option = st.session_state[SELECTION_KEYS[component_name]]
                st.session_state[component_widget_key(component_name, option, param_path)] = value
            else:
                st.session_state[base_widget_key(mapped)] = value

    st.session_state[TEMPLATE_OVERRIDES_KEY] = "\n".join(template.overrides)
    st.session_state["ui_raw_overrides"] = ""
    st.session_state[LOADED_TEMPLATE_KEY] = template.key


def collect_ui_state_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in st.session_state.items():
        if not key.startswith("ui_"):
            continue
        if key in {VIEW_KEY, SELECTED_RUN_KEY, CREATE_STEP_KEY}:
            continue
        snapshot[key] = value
    return snapshot


def restore_ui_state_snapshot(snapshot: dict[str, Any], defaults: dict[str, Any]) -> None:
    reset_form_to_defaults(defaults)
    for key, value in snapshot.items():
        if key.startswith("ui_"):
            st.session_state[key] = value


def render_card(label: str, value: str) -> None:
    st.markdown(
        (
            "<div class='fx-card'>"
            f"<div class='fx-card-label'>{label}</div>"
            f"<div class='fx-card-value'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def set_create_step(step: str) -> None:
    st.session_state[CREATE_STEP_KEY] = step


def queue_template_load(template_key: str, target_step: str = "run") -> None:
    st.session_state[PENDING_TEMPLATE_KEY] = template_key
    st.session_state[PENDING_STEP_KEY] = target_step


def queue_template_reset(target_step: str = "template") -> None:
    st.session_state[PENDING_RESET_KEY] = True
    st.session_state[PENDING_STEP_KEY] = target_step


def apply_pending_form_actions(
    defaults: dict[str, Any],
    templates: dict[str, TemplateSpec],
) -> None:
    pending_reset = bool(st.session_state.get(PENDING_RESET_KEY, False))
    pending_template = resolve_template_key(
        str(st.session_state.get(PENDING_TEMPLATE_KEY, "") or ""),
        templates,
    )
    pending_step = str(st.session_state.get(PENDING_STEP_KEY, "") or "")
    if not pending_reset and not pending_template and not pending_step:
        return

    if pending_reset:
        reset_form_to_defaults(defaults)
        st.session_state[TEMPLATE_PICKER_KEY] = ""

    if pending_template:
        template = templates.get(pending_template)
        if template is not None:
            reset_form_to_defaults(defaults)
            apply_template_to_state(template, defaults)
            st.session_state[TEMPLATE_PICKER_KEY] = pending_template

    if pending_step in CREATE_STEPS:
        st.session_state[CREATE_STEP_KEY] = pending_step

    st.session_state[PENDING_TEMPLATE_KEY] = ""
    st.session_state[PENDING_RESET_KEY] = False
    st.session_state[PENDING_STEP_KEY] = ""


def format_display_datetime(value: str | None) -> str:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return "-" if not value else str(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def get_main_flat_defaults(repo_root: Path) -> dict[str, Any]:
    main_config = get_main_config(repo_root)
    base_config = {
        key: value
        for key, value in main_config.items()
        if key not in {"defaults", "single_run_dir"}
    }
    return flatten_mapping(base_config)


def normalize_device_selection(raw_value: Any) -> list[int]:
    if isinstance(raw_value, list):
        values = raw_value
    elif raw_value in (None, ""):
        values = []
    else:
        try:
            values = yaml.safe_load(str(raw_value))
        except Exception:
            values = []
    if not isinstance(values, list):
        return []
    normalized: list[int] = []
    for value in values:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def render_select_input(
    label: str,
    options: list[str],
    *,
    key: str,
    none_label: str | None = None,
) -> None:
    if none_label is None:
        st.selectbox(label, options, key=key, format_func=lambda value: str(value))
        return
    st.selectbox(
        label,
        options,
        key=key,
        format_func=lambda value: none_label if not value else str(value),
    )


def resolve_template_key(raw_value: str, templates: dict[str, TemplateSpec]) -> str:
    if raw_value in templates:
        return raw_value
    for key, template in templates.items():
        if template.name == raw_value:
            return key
    return ""


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## FedXplore")
        if st.button("Dashboard", key="sidebar_dashboard", use_container_width=True):
            navigate_to(VIEW_DASHBOARD)
            rerun_app()
        if st.button("Create Run", key="sidebar_create_run", use_container_width=True):
            set_create_step(CREATE_STEPS[0])
            navigate_to(VIEW_CREATE)
            rerun_app()


def format_status_badge(status: str) -> str:
    normalized_status = "finished" if status == "unknown_finished" else status
    css_class = (
        normalized_status
        if normalized_status
        in {
            "running",
            "stopping",
            "stopped",
            "finished",
            "failed_to_start",
            "missing_status",
            "invalid_status",
            "missing_pid",
        }
        else "default"
    )
    return f"<span class='fx-status {css_class}'>{normalized_status}</span>"


def humanize_duration(started_at: str | None, finished_at: str | None, status: str) -> str:
    start_dt = parse_iso_datetime(started_at)
    if start_dt is None:
        return "-"
    end_dt = parse_iso_datetime(finished_at)
    if end_dt is None and status in {"running", "stopping"}:
        end_dt = datetime.now().astimezone()
    if end_dt is None:
        return "-"
    total_seconds = max(0, int((end_dt - start_dt).total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value not in (None, "", []):
            return str(value)
    return ""


def extract_run_meta(repo_root: Path, run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["run_dir"])
    spec = read_spec(run_dir)
    payload = spec.get("form_payload", {})
    selections = payload.get("selected_groups", {})
    base_params = payload.get("base_params", {})
    old_form_values = spec.get("form_values", {})

    dataset_name = first_non_empty(
        selections.get("train_dataset"),
        old_form_values.get("dataset"),
    )
    method_name = first_non_empty(
        selections.get("federated_method"),
        old_form_values.get("federated_method"),
    )
    logger_name = first_non_empty(
        selections.get("logger"),
        old_form_values.get("logger"),
    )
    rounds = first_non_empty(
        base_params.get("federated_params.communication_rounds"),
        old_form_values.get("communication_rounds"),
    )
    clients = first_non_empty(
        base_params.get("federated_params.amount_of_clients"),
        old_form_values.get("amount_of_clients"),
    )
    log_path = first_non_empty(
        spec.get("output_log_path"),
        run.get("stdout_path"),
    )

    return {
        "run_dir": run_dir,
        "spec": spec,
        "payload": payload,
        "name": run.get("run_name", run["run_id"]),
        "method": method_name or "-",
        "dataset": dataset_name or "-",
        "logger": logger_name or "-",
        "created_at": format_display_datetime(run.get("created_at", "")),
        "started_at": format_display_datetime(run.get("started_at", "")),
        "finished_at": format_display_datetime(run.get("finished_at", "")),
        "duration": humanize_duration(
            run.get("started_at"),
            run.get("finished_at"),
            run.get("status", ""),
        ),
        "status": "finished" if run.get("status") == "unknown_finished" else run.get("status", "unknown"),
        "rounds": rounds or "-",
        "clients": clients or "-",
        "pid": run.get("pid", "-"),
        "mlflow_url": run.get("mlflow_url", ""),
        "log_path": format_rel_path(repo_root, log_path),
    }


def format_rel_path(repo_root: Path, path_value: str | Path) -> str:
    path = Path(path_value)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def render_dashboard_page(repo_root: Path, runs: list[dict[str, Any]]) -> None:
    st.title("FedXplore Dashboard")

    total_runs = len(runs)
    running_runs = sum(1 for run in runs if run.get("status") == "running")
    stopping_runs = sum(1 for run in runs if run.get("status") == "stopping")
    cards = st.columns(3)
    with cards[0]:
        render_card("Runs", str(total_runs))
    with cards[1]:
        render_card("Running", str(running_runs))
    with cards[2]:
        render_card("Stopping", str(stopping_runs))

    st.markdown("### Runs")
    if not runs:
        st.write("No runs.")
        return

    metas = [extract_run_meta(repo_root, run) for run in runs]
    method_options = ["All"] + sorted({meta["method"] for meta in metas if meta["method"] not in {"", "-"}})
    dataset_options = ["All"] + sorted({meta["dataset"] for meta in metas if meta["dataset"] not in {"", "-"}})
    status_options = ["All"] + sorted({meta["status"] for meta in metas if meta["status"]})

    with st.container(border=True):
        filter_cols = st.columns([1.9, 1.2, 1.2, 1.1])
        with filter_cols[0]:
            st.text_input("Name", key=DASHBOARD_NAME_FILTER_KEY, placeholder="Search run")
        with filter_cols[1]:
            st.selectbox("Method", method_options, key=DASHBOARD_METHOD_FILTER_KEY)
        with filter_cols[2]:
            st.selectbox("Dataset", dataset_options, key=DASHBOARD_DATASET_FILTER_KEY)
        with filter_cols[3]:
            st.selectbox("Status", status_options, key=DASHBOARD_STATUS_FILTER_KEY)

        filtered_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        name_filter = st.session_state.get(DASHBOARD_NAME_FILTER_KEY, "").strip().lower()
        method_filter = st.session_state.get(DASHBOARD_METHOD_FILTER_KEY, "All")
        dataset_filter = st.session_state.get(DASHBOARD_DATASET_FILTER_KEY, "All")
        status_filter = st.session_state.get(DASHBOARD_STATUS_FILTER_KEY, "All")
        for run, meta in zip(runs, metas):
            if name_filter and name_filter not in meta["name"].lower():
                continue
            if method_filter != "All" and meta["method"] != method_filter:
                continue
            if dataset_filter != "All" and meta["dataset"] != dataset_filter:
                continue
            if status_filter != "All" and meta["status"] != status_filter:
                continue
            filtered_rows.append((run, meta))

        st.markdown("<div class='fx-divider'></div>", unsafe_allow_html=True)
        header_cols = st.columns([2.5, 1.25, 1.25, 1.35, 1.0, 0.95, 0.8])
        header_labels = ["Name", "Method", "Dataset", "Created", "Time", "Status", ""]
        for col, label in zip(header_cols, header_labels):
            with col:
                st.markdown(f"<div class='fx-table-header'>{label}</div>", unsafe_allow_html=True)

        st.markdown("<div class='fx-divider'></div>", unsafe_allow_html=True)

        if not filtered_rows:
            st.caption("No runs match the current filters.")
            return

        for index, (run, meta) in enumerate(filtered_rows):
            row_cols = st.columns([2.5, 1.25, 1.25, 1.35, 1.0, 0.95, 0.8])
            with row_cols[0]:
                st.markdown(f"**{meta['name']}**")
            with row_cols[1]:
                st.write(meta["method"])
            with row_cols[2]:
                st.write(meta["dataset"])
            with row_cols[3]:
                st.write(meta["created_at"])
            with row_cols[4]:
                st.write(meta["duration"])
            with row_cols[5]:
                st.markdown(format_status_badge(meta["status"]), unsafe_allow_html=True)
            with row_cols[6]:
                if st.button("Open", key=f"open_run_{run['run_id']}", use_container_width=True):
                    navigate_to(VIEW_RUN, run_id=run["run_id"])
                    rerun_app()
            if index < len(filtered_rows) - 1:
                st.markdown("<div class='fx-divider'></div>", unsafe_allow_html=True)


def render_value_widget(label: str, key: str, default_value: Any, *, height: int = 88) -> None:
    seed_state_value(key, default_value)
    if isinstance(default_value, bool):
        st.checkbox(label, key=key)
    elif isinstance(default_value, int) and not isinstance(default_value, bool):
        st.number_input(label, key=key, step=1)
    elif isinstance(default_value, float):
        st.number_input(label, key=key, format="%.6f")
    elif isinstance(default_value, str) and len(default_value) < 80 and "\n" not in default_value:
        st.text_input(label, key=key)
    else:
        st.text_area(label, key=key, height=height)


def render_flat_param_editors(
    section_title: str,
    flat_defaults: dict[str, Any],
    *,
    widget_key_builder,
    columns: int = 2,
) -> None:
    if not flat_defaults:
        return
    if section_title:
        st.markdown(f"#### {section_title}")
    paths = list(flat_defaults.keys())
    for chunk_start in range(0, len(paths), columns):
        cols = st.columns(columns)
        for column_index, path in enumerate(paths[chunk_start : chunk_start + columns]):
            default_value = flat_defaults[path]
            widget_key = widget_key_builder(path)
            with cols[column_index]:
                render_value_widget(path, widget_key, default_value)


def collect_flat_param_values(
    flat_defaults: dict[str, Any],
    *,
    widget_key_builder,
) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    errors: list[str] = []
    for path, default_value in flat_defaults.items():
        widget_key = widget_key_builder(path)
        raw_value = st.session_state.get(widget_key, widget_seed_value(default_value))
        try:
            values[path] = parse_widget_value(raw_value, default_value)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return values, errors


def get_component_flat_defaults(repo_root: Path, component_name: str, option: str) -> dict[str, Any]:
    if component_name == "preaggregator" and not option:
        return {}
    if component_name == "trust_dataset":
        return {}
    return get_component_default_params(repo_root, component_name, option)


def render_component_section(repo_root: Path, component_name: str, option: str) -> None:
    flat_defaults = get_component_flat_defaults(repo_root, component_name, option)
    if not flat_defaults:
        return
    st.markdown(f"##### {COMPONENT_LABELS[component_name]} parameters")
    render_flat_param_editors(
        "",
        flat_defaults,
        widget_key_builder=lambda path: component_widget_key(component_name, option, path),
    )


def collect_component_values(
    repo_root: Path, component_name: str, option: str
) -> tuple[dict[str, Any], list[str]]:
    flat_defaults = get_component_flat_defaults(repo_root, component_name, option)
    if not flat_defaults:
        return {}, []
    return collect_flat_param_values(
        flat_defaults,
        widget_key_builder=lambda path: component_widget_key(component_name, option, path),
    )


def render_gpu_monitor() -> None:
    def render_body() -> None:
        gpu_rows = query_gpus()
        if not gpu_rows:
            st.caption("nvidia-smi is not available or no NVIDIA GPU detected")
            return
        for row in gpu_rows:
            used = row.get("memory used MiB", 0)
            total = row.get("memory total MiB", 1)
            util = row.get("utilization %", 0)
            mem_percent = 0 if not total else min(100, int((used / total) * 100))
            st.markdown(
                (
                    "<div class='fx-gpu-panel'>"
                    f"<div class='fx-gpu-head'><span>GPU {row['index']} · {row['name']}</span>"
                    f"<span>{used}/{total} MiB · {util}%</span></div>"
                    f"<div class='fx-gpu-bar'><span style='width:{mem_percent}%'></span></div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    fragment_api = getattr(st, "fragment", None)
    if callable(fragment_api):
        fragment_api(run_every="2s")(render_body)()
        return
    render_body()
    if st.button("Refresh GPU", key="gpu_refresh_button", use_container_width=True):
        rerun_app()


def collect_form_payload(
    repo_root: Path,
) -> tuple[dict[str, Any], list[str], list[str], str, list[str], str]:
    selected_groups = {
        name: st.session_state[state_key]
        for name, state_key in SELECTION_KEYS.items()
    }

    main_config = get_main_config(repo_root)
    base_config = {
        key: value
        for key, value in main_config.items()
        if key not in {"defaults", "single_run_dir"}
    }
    base_defaults = flatten_mapping(base_config)
    base_params, base_errors = collect_flat_param_values(
        base_defaults,
        widget_key_builder=base_widget_key,
    )
    device_mode = str(st.session_state.get(DEVICE_MODE_KEY, "cuda") or "cuda")
    device_ids = normalize_device_selection(st.session_state.get(DEVICE_IDS_KEY, []))
    base_params["training_params.device"] = device_mode
    base_params["training_params.device_ids"] = device_ids if device_mode == "cuda" else []
    attack_type = str(st.session_state.get(ATTACK_TYPE_KEY, "no_attack") or "no_attack")
    base_params["federated_params.clients_attack_types"] = attack_type
    attack_params: dict[str, Any] = {}
    attack_errors: list[str] = []

    component_params: dict[str, dict[str, Any]] = {}
    component_errors: list[str] = []
    for component_name in COMPONENT_LABELS:
        option = selected_groups.get(component_name, "")
        values, errors = collect_component_values(repo_root, component_name, option)
        component_params[component_name] = values
        component_errors.extend(
            f"{COMPONENT_LABELS[component_name]} / {error}" for error in errors
        )

    raw_override_text = st.session_state.get("ui_raw_overrides", "")
    raw_errors: list[str] = []
    raw_overrides: list[str] = []
    try:
        raw_overrides = parse_raw_overrides(raw_override_text)
    except ValueError as exc:
        raw_errors.append(str(exc))

    template_override_text = st.session_state.get(TEMPLATE_OVERRIDES_KEY, "")
    template_errors: list[str] = []
    template_overrides: list[str] = []
    try:
        template_overrides = parse_raw_overrides(template_override_text)
    except ValueError as exc:
        template_errors.append(str(exc))

    form_payload = {
        "run_name": st.session_state["ui_run_name"].strip() or DEFAULT_RUN_NAME,
        "selected_groups": selected_groups,
        "base_params": base_params,
        "component_params": component_params,
        "attack_type": attack_type,
        "attack_params": attack_params,
        "disable_proxy_for_mlflow": True,
    }
    errors = [*base_errors, *attack_errors, *component_errors, *raw_errors, *template_errors]
    return (
        form_payload,
        raw_overrides,
        errors,
        raw_override_text,
        template_overrides,
        template_override_text,
    )


def render_template_section(
    defaults: dict[str, Any],
    templates: dict[str, TemplateSpec],
) -> None:
    st.markdown("### Template")
    st.markdown("<div class='fx-step-note'>Choose a template or keep a manual setup.</div>", unsafe_allow_html=True)
    template_options = [""] + list(templates.keys())
    def format_template_value(value: str) -> str:
        if not value:
            return "Manual"
        template = templates.get(str(value))
        return template.name if template is not None else str(value)
    selected_template = st.selectbox(
        "Template",
        options=template_options,
        key=TEMPLATE_PICKER_KEY,
        format_func=format_template_value,
    )
    resolved_template_key = resolve_template_key(str(selected_template or ""), templates)
    action_cols = st.columns([1, 1, 5])
    with action_cols[0]:
        if st.button(
            "Load",
            key="template_load_button",
            disabled=not resolved_template_key,
            use_container_width=True,
        ):
            queue_template_load(resolved_template_key, target_step="run")
            rerun_app()
    with action_cols[1]:
        if st.button("Reset", key="template_reset_button", use_container_width=True):
            queue_template_reset(target_step="template")
            rerun_app()
    with action_cols[2]:
        loaded_template = st.session_state.get(LOADED_TEMPLATE_KEY, "")
        if loaded_template:
            st.write(templates[loaded_template].description or templates[loaded_template].name)
        elif resolved_template_key:
            st.write(
                templates[resolved_template_key].description
                or templates[resolved_template_key].name
            )


def render_run_setup_step(repo_root: Path) -> None:
    st.markdown("### Run")
    main_flat_defaults = get_main_flat_defaults(repo_root)
    general_left, general_right = st.columns([1.6, 1.4])
    with general_left:
        st.text_input("Run name", key="ui_run_name")
        render_value_widget(
            "random_state",
            base_widget_key("random_state"),
            main_flat_defaults.get("random_state", 42),
        )
    with general_right:
        log_path = preview_stdout_path(repo_root, st.session_state["ui_run_name"])
        st.text_input("Primary log file", value=format_rel_path(repo_root, log_path), disabled=True)


def render_data_setup_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Setup")
    dataset_cols = st.columns(3)
    with dataset_cols[0]:
        render_select_input("Train dataset", options["dataset"], key="ui_train_dataset")
    with dataset_cols[1]:
        render_select_input("Test dataset", options["dataset"], key="ui_test_dataset")
    with dataset_cols[2]:
        render_select_input(
            "Trust dataset",
            [""] + options["dataset"],
            key="ui_trust_dataset",
            none_label="None",
        )

    main_flat_defaults = get_main_flat_defaults(repo_root)
    render_flat_param_editors(
        "Base setup",
        {path: main_flat_defaults[path] for path in SETUP_BASE_PATHS},
        widget_key_builder=base_widget_key,
    )

    setup_component_cols = st.columns(3)
    setup_components = [
        ("distribution", "ui_distribution", options["distribution"]),
        ("model", "ui_model", options["model"]),
        ("model_trainer", "ui_model_trainer", options["model_trainer"]),
    ]
    for column, (component_name, state_key, component_options) in zip(
        setup_component_cols, setup_components
    ):
        with column:
            render_select_input(
                COMPONENT_LABELS[component_name],
                component_options,
                key=state_key,
            )
            render_component_section(repo_root, component_name, st.session_state[state_key])


def render_method_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Method")
    method_cols = st.columns(3)
    method_components = [
        ("federated_method", "ui_federated_method", options["federated_method"]),
        ("client_selector", "ui_client_selector", options["client_selector"]),
        ("preaggregator", "ui_preaggregator", options["preaggregator"]),
    ]
    for column, (component_name, state_key, component_options) in zip(
        method_cols, method_components
    ):
        with column:
            render_select_input(
                COMPONENT_LABELS[component_name],
                component_options,
                key=state_key,
                none_label="None" if component_name == "preaggregator" else None,
            )
            render_component_section(repo_root, component_name, st.session_state[state_key])


def render_logging_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Logging")
    logging_cols = st.columns([1.2, 2])
    with logging_cols[0]:
        render_select_input("Logger", options["logger"], key="ui_logger")
    with logging_cols[1]:
        render_component_section(repo_root, "logger", st.session_state["ui_logger"])
        if st.session_state["ui_logger"] == "mlflow":
            tracking_uri = st.session_state.get(
                component_widget_key("logger", "mlflow", "tracking_uri"),
                "",
            )
            if not st.session_state.get("ui_mlflow_ui_url") and tracking_uri:
                st.session_state["ui_mlflow_ui_url"] = normalize_mlflow_ui_url(str(tracking_uri))
            st.text_input("MLflow UI", key="ui_mlflow_ui_url")


def render_training_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Training & Other")
    main_flat_defaults = get_main_flat_defaults(repo_root)
    base_cols = st.columns(2)
    with base_cols[0]:
        render_select_input("Optimizer", options["optimizer"], key="ui_optimizer")
        render_component_section(repo_root, "optimizer", st.session_state["ui_optimizer"])
        render_flat_param_editors(
            "Federated",
            {path: main_flat_defaults[path] for path in OTHER_BASE_PATHS},
            widget_key_builder=base_widget_key,
        )
    with base_cols[1]:
        render_select_input("Loss", options["loss"], key="ui_loss")
        render_component_section(repo_root, "loss", st.session_state["ui_loss"])


def render_attacks_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Attacks")
    main_flat_defaults = get_main_flat_defaults(repo_root)
    render_select_input("Attack type", options["attack_type"], key=ATTACK_TYPE_KEY)
    render_select_input(
        "Attack scheme",
        [
            "no_attack",
            "constant",
            "random_rounds",
            "random_clients",
            "random_rounds_random_clients",
        ],
        key=base_widget_key("federated_params.attack_scheme"),
    )
    render_flat_param_editors(
        "Attack schedule",
        {
            path: main_flat_defaults[path]
            for path in [
                "federated_params.prop_attack_clients",
                "federated_params.prop_attack_rounds",
            ]
        },
        widget_key_builder=base_widget_key,
    )


def render_technical_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Technical")
    gpu_rows = query_gpus()
    gpu_options = [int(row["index"]) for row in gpu_rows if isinstance(row.get("index"), int)]
    if DEVICE_MODE_KEY not in st.session_state:
        st.session_state[DEVICE_MODE_KEY] = "cuda"
    if DEVICE_IDS_KEY not in st.session_state:
        st.session_state[DEVICE_IDS_KEY] = gpu_options[:1] if gpu_options else []
    current_device_ids = normalize_device_selection(st.session_state.get(DEVICE_IDS_KEY, []))
    st.session_state[DEVICE_IDS_KEY] = [device_id for device_id in current_device_ids if device_id in gpu_options]
    technical_cols = st.columns([1.4, 1.1, 1.7])
    with technical_cols[0]:
        render_select_input("Manager", options["manager"], key="ui_manager")
        render_component_section(repo_root, "manager", st.session_state["ui_manager"])
    with technical_cols[1]:
        render_select_input(
            "Batch generator",
            options["manager_batch_generator"],
            key="ui_manager_batch_generator",
        )
        render_component_section(
            repo_root,
            "manager_batch_generator",
            st.session_state["ui_manager_batch_generator"],
        )
        st.radio(
            "Device",
            options=["cpu", "cuda"],
            key=DEVICE_MODE_KEY,
            horizontal=True,
        )
        if st.session_state[DEVICE_MODE_KEY] == "cuda":
            st.multiselect(
                "GPU device_ids",
                options=gpu_options,
                key=DEVICE_IDS_KEY,
                format_func=lambda value: f"GPU {value}",
            )
        else:
            st.session_state[DEVICE_IDS_KEY] = []
    with technical_cols[2]:
        render_gpu_monitor()


def render_launch_step(repo_root: Path) -> None:
    st.markdown("### Launch")
    template_override_text = st.session_state.get(TEMPLATE_OVERRIDES_KEY, "")
    if template_override_text.strip():
        st.text_area(
            "Template overrides",
            key=TEMPLATE_OVERRIDES_KEY,
            height=140,
        )
    st.text_area(
        "Raw Hydra overrides",
        key="ui_raw_overrides",
        height=180,
    )

    (
        form_payload,
        raw_overrides,
        errors,
        raw_override_text,
        template_overrides,
        template_override_text,
    ) = collect_form_payload(repo_root)
    if errors:
        st.error("\n".join(errors))
        run_disabled = True
        overrides: list[str] = []
    else:
        structured_and_user_overrides = build_overrides(form_payload, raw_overrides)
        overrides = [*template_overrides, *structured_and_user_overrides]
        duplicates = find_duplicate_override_keys(overrides)
        cmd = build_command(repo_root, overrides)
        manual_command = format_manual_shell_command(cmd, preview_stdout_path(repo_root, form_payload["run_name"]))
        st.code(manual_command, language="bash")
        if duplicates:
            st.caption("Duplicate override keys: " + ", ".join(duplicates))
        run_disabled = False

    nav_cols = st.columns([1, 1, 5])
    with nav_cols[0]:
        st.button(
            "Back",
            key="create_prev_launch",
            on_click=set_create_step,
            args=[CREATE_STEPS[-2]],
            use_container_width=True,
        )
    with nav_cols[1]:
        run_clicked = st.button(
            "Run",
            key="run_button",
            disabled=run_disabled,
            use_container_width=True,
        )

    if run_clicked:
        subprocess_env = None
        bypass_hosts: list[str] = []
        mlflow_url = ""
        if form_payload["selected_groups"]["logger"] == "mlflow":
            logger_params = form_payload["component_params"]["logger"]
            tracking_uri = str(logger_params.get("tracking_uri", "")).strip()
            mlflow_url = normalize_mlflow_ui_url(
                st.session_state.get("ui_mlflow_ui_url", "") or tracking_uri
            )
            subprocess_env, bypass_hosts = build_subprocess_env(
                disable_proxy=True,
                mlflow_tracking_uri=tracking_uri,
            )

        status = start_run(
            repo_root=repo_root,
            run_name=form_payload["run_name"],
            overrides=overrides,
            mlflow_url=mlflow_url or None,
            subprocess_env=subprocess_env,
            spec_data={
                "form_payload": form_payload,
                "ui_state_snapshot": collect_ui_state_snapshot(),
                "raw_overrides_text": raw_override_text,
                "template_overrides_text": template_override_text,
                "proxy_bypass_hosts": bypass_hosts,
            },
        )
        navigate_to(VIEW_RUN, run_id=status["run_id"])
        rerun_app()


def render_create_stepper() -> None:
    st.radio(
        "Create step",
        options=CREATE_STEPS,
        key=CREATE_STEP_KEY,
        horizontal=True,
        label_visibility="collapsed",
        format_func=lambda step: CREATE_STEP_LABELS[step],
    )


def render_step_navigation(current_step: str) -> None:
    current_index = CREATE_STEPS.index(current_step)
    if current_step == "launch":
        return

    nav_cols = st.columns([1, 1, 5])
    with nav_cols[0]:
        st.button(
            "Back",
            key=f"create_prev_{current_step}",
            disabled=current_index == 0,
            on_click=set_create_step,
            args=[CREATE_STEPS[max(0, current_index - 1)]],
            use_container_width=True,
        )
    with nav_cols[1]:
        st.button(
            "Next",
            key=f"create_next_{current_step}",
            disabled=current_index == len(CREATE_STEPS) - 1,
            on_click=set_create_step,
            args=[CREATE_STEPS[min(len(CREATE_STEPS) - 1, current_index + 1)]],
            use_container_width=True,
        )


def render_create_page(
    repo_root: Path,
    defaults: dict[str, Any],
    options: dict[str, list[str]],
    templates: dict[str, TemplateSpec],
) -> None:
    apply_pending_form_actions(defaults, templates)

    top_cols = st.columns([5, 1.4])
    with top_cols[0]:
        st.title("Create Run")
    with top_cols[1]:
        if st.button("Dashboard", key="create_to_dashboard", use_container_width=True):
            navigate_to(VIEW_DASHBOARD)
            rerun_app()

    render_create_stepper()
    current_step = st.session_state.get(CREATE_STEP_KEY, CREATE_STEPS[0])

    if current_step == "template":
        render_template_section(defaults, templates)
    elif current_step == "run":
        render_run_setup_step(repo_root)
    elif current_step == "setup":
        render_data_setup_step(repo_root, options)
    elif current_step == "method":
        render_method_step(repo_root, options)
    elif current_step == "logging":
        render_logging_step(repo_root, options)
    elif current_step == "training":
        render_training_step(repo_root, options)
    elif current_step == "attacks":
        render_attacks_step(repo_root, options)
    elif current_step == "technical":
        render_technical_step(repo_root, options)
    else:
        render_launch_step(repo_root)

    render_step_navigation(current_step)


def render_kv_table(rows: list[tuple[str, Any]]) -> None:
    rendered = []
    for label, value in rows:
        rendered.append(
            f"<div class='fx-kv-label'>{label}</div><div>{value if value not in (None, '') else '-'}</div>"
        )
    st.markdown("<div class='fx-kv'>" + "".join(rendered) + "</div>", unsafe_allow_html=True)


def render_parameters_view(repo_root: Path, run: dict[str, Any], meta: dict[str, Any]) -> None:
    payload = meta["payload"]
    if not payload:
        st.write("Structured parameters were not stored for this run.")
        return

    selections = payload.get("selected_groups", {})
    base_params = payload.get("base_params", {})
    component_params = payload.get("component_params", {})
    attack_type = payload.get("attack_type", "no_attack")
    tabs = st.tabs([name for name, _ in PARAMETER_TAB_COMPONENTS] + ["Raw"])
    for tab, (tab_name, components) in zip(tabs, PARAMETER_TAB_COMPONENTS):
        with tab:
            if tab_name == "Setup":
                render_kv_table(
                    [
                        ("Train dataset", selections.get("train_dataset")),
                        ("Test dataset", selections.get("test_dataset")),
                        ("Trust dataset", selections.get("trust_dataset") or "None"),
                        ("batch_size", base_params.get("training_params.batch_size")),
                        ("num_workers", base_params.get("training_params.num_workers")),
                        ("amount_of_clients", base_params.get("federated_params.amount_of_clients")),
                        ("client_subset_size", base_params.get("federated_params.client_subset_size")),
                        ("communication_rounds", base_params.get("federated_params.communication_rounds")),
                        ("local_epochs", base_params.get("federated_params.local_epochs")),
                        ("client_train_val_prop", base_params.get("federated_params.client_train_val_prop")),
                    ]
                )
            elif tab_name == "Method":
                render_kv_table(
                    [
                        ("Method", selections.get("federated_method")),
                        ("Client selector", selections.get("client_selector")),
                        ("Preaggregator", selections.get("preaggregator") or "None"),
                    ]
                )
            elif tab_name == "Logging":
                render_kv_table(
                    [
                        ("Logger", selections.get("logger")),
                        ("MLflow", meta.get("mlflow_url") or "-"),
                    ]
                )
            elif tab_name == "Technical":
                render_kv_table(
                    [
                        ("Manager", selections.get("manager")),
                        ("Batch generator", selections.get("manager_batch_generator")),
                        ("device", base_params.get("training_params.device")),
                        ("device_ids", dump_complex_value(base_params.get("training_params.device_ids"))),
                    ]
                )
            elif tab_name == "Attacks":
                render_kv_table(
                    [
                        ("Attack type", attack_type),
                        ("Attack scheme", base_params.get("federated_params.attack_scheme")),
                        ("prop_attack_clients", base_params.get("federated_params.prop_attack_clients")),
                        ("prop_attack_rounds", base_params.get("federated_params.prop_attack_rounds")),
                    ]
                )

            for component_name in components:
                if component_name not in component_params:
                    continue
                values = component_params.get(component_name, {})
                if not values:
                    continue
                st.markdown(f"#### {COMPONENT_LABELS[component_name]}")
                st.code(
                    yaml.safe_dump(
                        unflatten_mapping(values),
                        sort_keys=False,
                        allow_unicode=False,
                    ),
                    language="yaml",
                )

            if tab_name == "Training":
                st.markdown("#### Base")
                training_base = {
                    key: value
                    for key, value in base_params.items()
                    if key in OTHER_BASE_PATHS
                }
                st.code(
                    yaml.safe_dump(
                        unflatten_mapping(training_base),
                        sort_keys=False,
                        allow_unicode=False,
                    ),
                    language="yaml",
                )

    with tabs[-1]:
        spec = read_spec(Path(run["run_dir"]))
        template_overrides_text = spec.get("template_overrides_text", "")
        if template_overrides_text:
            st.text_area(
                "Template overrides",
                value=template_overrides_text,
                height=160,
                disabled=True,
            )
        st.text_area(
            "Raw overrides",
            value=spec.get("raw_overrides_text", ""),
            height=160,
            disabled=True,
        )


def render_journal_view(run: dict[str, Any]) -> None:
    events = read_run_events(Path(run["run_dir"]))
    if not events:
        st.write("No journal entries.")
        return
    rows = [
        {
            "timestamp": format_display_datetime(event.get("timestamp", "")),
            "event": event.get("event_type", ""),
            "message": event.get("message", ""),
        }
        for event in reversed(events)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_logs_view(run: dict[str, Any]) -> None:
    controls = st.columns([1.2, 1, 3.4])
    with controls[0]:
        st.number_input("Lines", min_value=20, step=20, key=LAST_LINES_KEY)
    with controls[1]:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("Refresh", key="run_log_refresh", use_container_width=True):
            rerun_app()
    line_count = int(st.session_state[LAST_LINES_KEY])
    log_path = Path(run.get("stdout_path", ""))
    st.text_area(
        "Log",
        value=tail_file(log_path, line_count),
        height=520,
        disabled=True,
    )


def render_files_view(repo_root: Path, run: dict[str, Any]) -> None:
    run_dir = Path(run["run_dir"])
    spec_path = run_dir / "spec.yaml"
    status_path = run_dir / "status.json"
    command_path = run_dir / "command.sh"
    st.code(command_path.read_text(encoding="utf-8"), language="bash")
    st.code(spec_path.read_text(encoding="utf-8"), language="yaml")
    st.code(status_path.read_text(encoding="utf-8"), language="json")
    st.write(format_rel_path(repo_root, run_dir))


def render_overview_view(repo_root: Path, run: dict[str, Any], meta: dict[str, Any]) -> None:
    render_kv_table(
        [
            ("Run id", run["run_id"]),
            ("Method", meta["method"]),
            ("Dataset", meta["dataset"]),
            ("Logger", meta["logger"]),
            ("PID", meta["pid"]),
            ("Created", meta["created_at"] or "-"),
            ("Started", meta["started_at"] or "-"),
            ("Finished", meta["finished_at"] or "-"),
            ("Time", meta["duration"]),
            ("Log file", meta["log_path"]),
        ]
    )
    if meta["mlflow_url"]:
        st.markdown(f"[MLflow]({meta['mlflow_url']})")


def render_run_header(meta: dict[str, Any], run: dict[str, Any]) -> None:
    st.markdown(
        (
            "<div class='fx-detail-hero'>"
            f"<div class='fx-detail-title'>{meta['name']}</div>"
            f"<div class='fx-detail-subtitle'>{run['run_id']}</div>"
            f"<div style='margin-top:0.45rem'>{format_status_badge(meta['status'])}</div>"
            "<div class='fx-detail-grid'>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>Method</div><div class='fx-detail-value'>{meta['method']}</div></div>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>Dataset</div><div class='fx-detail-value'>{meta['dataset']}</div></div>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>Created</div><div class='fx-detail-value'>{meta['created_at']}</div></div>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>Duration</div><div class='fx-detail-value'>{meta['duration']}</div></div>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>Logger</div><div class='fx-detail-value'>{meta['logger']}</div></div>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>PID</div><div class='fx-detail-value'>{meta['pid']}</div></div>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>Log file</div><div class='fx-detail-value'>{meta['log_path']}</div></div>"
            f"<div class='fx-detail-item'><div class='fx-detail-label'>Started</div><div class='fx-detail-value'>{meta['started_at']}</div></div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_run_detail_page(repo_root: Path, runs: list[dict[str, Any]], defaults: dict[str, Any]) -> None:
    run_id = st.session_state.get(SELECTED_RUN_KEY, "")
    run_map = {run["run_id"]: run for run in runs}
    run = run_map.get(run_id)
    if run is None:
        navigate_to(VIEW_DASHBOARD)
        rerun_app()
        return

    meta = extract_run_meta(repo_root, run)
    header_cols = st.columns([4.8, 2.8])
    with header_cols[0]:
        render_run_header(meta, run)
    with header_cols[1]:
        action_grid = st.columns(2)
        with action_grid[0]:
            if st.button("Dashboard", key="run_to_dashboard", use_container_width=True):
                navigate_to(VIEW_DASHBOARD)
                rerun_app()
        with action_grid[1]:
            can_stop = run.get("status") == "running"
            if st.button("Stop", key="run_stop", disabled=not can_stop, use_container_width=True):
                stop_run(Path(run["run_dir"]))
                rerun_app()
        with action_grid[0]:
            if st.button("Clone", key="run_rerun", use_container_width=True):
                spec = meta["spec"]
                snapshot = spec.get("ui_state_snapshot")
                if snapshot:
                    restore_ui_state_snapshot(snapshot, defaults)
                set_create_step("run")
                navigate_to(VIEW_CREATE)
                rerun_app()
        with action_grid[1]:
            if meta["mlflow_url"]:
                if hasattr(st, "link_button"):
                    st.link_button("MLflow", meta["mlflow_url"], use_container_width=True)
                else:
                    st.markdown(f"[MLflow]({meta['mlflow_url']})")
            else:
                st.button("MLflow", key="run_mlflow_disabled", disabled=True, use_container_width=True)

    tabs = st.tabs(["Logs", "Parameters", "Journal", "Files", "Overview"])
    with tabs[0]:
        render_logs_view(run)
    with tabs[1]:
        render_parameters_view(repo_root, run, meta)
    with tabs[2]:
        render_journal_view(run)
    with tabs[3]:
        render_files_view(repo_root, run)
    with tabs[4]:
        render_overview_view(repo_root, run, meta)


def main() -> None:
    st.set_page_config(
        page_title="FedXplore Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    install_keyboard_guard()
    apply_page_styles()

    try:
        repo_root = get_repo_root(Path(__file__).resolve())
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    options = get_option_sets(repo_root)
    defaults = build_default_state(repo_root, options)
    ensure_state_defaults(defaults)
    keep_ui_state_alive()
    restore_view_from_query_params()

    try:
        templates = load_templates(repo_root / "ui/templates")
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar()
    runs = list_runs(repo_root)
    view = st.session_state.get(VIEW_KEY, VIEW_DASHBOARD)
    if view == VIEW_CREATE:
        render_create_page(repo_root, defaults, options, templates)
    elif view == VIEW_RUN:
        render_run_detail_page(repo_root, runs, defaults)
    else:
        render_dashboard_page(repo_root, runs)


if __name__ == "__main__":
    main()
