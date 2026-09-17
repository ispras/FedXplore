from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

try:
    from .analytics import (
        load_final_metrics,
        load_metric_histories,
        metric_names as available_metric_names,
        metric_points_frame,
    )
    from .artifacts import (
        ArtifactDownload,
        ArtifactKind,
        ArtifactMetadata,
        build_artifact_preview,
        download_artifact,
        list_run_artifacts,
    )
    from .comparison import (
        build_config_diff,
        comparison_has_active_runs,
        experiment_config_from_spec,
    )
    from .provenance import load_provenance
    from .styles import inject_global_styles, render_final_metrics_table, render_metric_chart_card
    from .create_run import build_experiment_summary, initial_dataset_roles, readable_label, readable_option, validate_experiment_state
    from .research_catalog import load_research_catalog, metadata_for, ordered_options
    from .examples import (
        ExampleDefinition,
        available_preferred_metrics,
        build_example_launch_plan,
        build_group_id,
        example_context_from_specs,
        launch_example_suite,
        load_examples,
        ordered_examples,
    )
except ImportError:  # pragma: no cover - supports `streamlit run ui/run_ui.py`
    ui_dir = str(Path(__file__).resolve().parent)
    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    from analytics import (
        load_final_metrics,
        load_metric_histories,
        metric_names as available_metric_names,
        metric_points_frame,
    )
    from artifacts import (
        ArtifactDownload,
        ArtifactKind,
        ArtifactMetadata,
        build_artifact_preview,
        download_artifact,
        list_run_artifacts,
    )
    from comparison import (
        build_config_diff,
        comparison_has_active_runs,
        experiment_config_from_spec,
    )
    from provenance import load_provenance
    from styles import inject_global_styles, render_final_metrics_table, render_metric_chart_card
    from create_run import build_experiment_summary, initial_dataset_roles, readable_label, readable_option, validate_experiment_state
    from research_catalog import load_research_catalog, metadata_for, ordered_options
    from examples import (
        ExampleDefinition,
        available_preferred_metrics,
        build_example_launch_plan,
        build_group_id,
        example_context_from_specs,
        launch_example_suite,
        load_examples,
        ordered_examples,
    )

try:
    from .launcher import (
        DEFAULT_RUN_NAME,
        DEFAULT_LOCAL_MLFLOW_UI_URL,
        TemplateSpec,
        build_command,
        build_mlflow_run_url,
        build_overrides,
        build_subprocess_env,
        discover_config_group_options,
        ensure_local_mlflow_ui,
        extract_main_default_selections,
        find_duplicate_override_keys,
        flatten_mapping,
        format_manual_shell_command,
        get_component_default_params,
        get_local_mlflow_tracking_uri,
        get_main_config,
        get_mlflow_defaults,
        get_repo_root,
        infer_mlflow_target,
        list_runs,
        load_templates,
        normalize_mlflow_ui_url,
        parse_iso_datetime,
        parse_raw_overrides,
        persist_mlflow_metadata,
        preview_stdout_path,
        query_gpus,
        read_run_events,
        read_status,
        read_spec,
        rerun_saved_run,
        start_run,
        stop_run,
        tail_file,
        unflatten_mapping,
    )
except ImportError:
    from launcher import (
        DEFAULT_RUN_NAME,
        DEFAULT_LOCAL_MLFLOW_UI_URL,
        TemplateSpec,
        build_command,
        build_mlflow_run_url,
        build_overrides,
        build_subprocess_env,
        discover_config_group_options,
        ensure_local_mlflow_ui,
        extract_main_default_selections,
        find_duplicate_override_keys,
        flatten_mapping,
        format_manual_shell_command,
        get_component_default_params,
        get_local_mlflow_tracking_uri,
        get_main_config,
        get_mlflow_defaults,
        get_repo_root,
        infer_mlflow_target,
        list_runs,
        load_templates,
        normalize_mlflow_ui_url,
        parse_iso_datetime,
        parse_raw_overrides,
        persist_mlflow_metadata,
        preview_stdout_path,
        query_gpus,
        read_run_events,
        read_status,
        read_spec,
        rerun_saved_run,
        start_run,
        stop_run,
        tail_file,
        unflatten_mapping,
    )


VIEW_KEY = "ui_view"
VIEW_DASHBOARD = "dashboard"
VIEW_CREATE = "create_run"
VIEW_RUN = "run_detail"
VIEW_COMPARE = "compare"
VIEW_EXAMPLES = "examples"

SELECTED_RUN_KEY = "ui_selected_run_id"
COMPARE_RUN_IDS_KEY = "ui_compare_run_ids"
COMPARE_PICKER_KEY = "ui_compare_picker"
COMPARE_ADD_RUN_KEY = "ui_compare_add_run"
COMPARE_ADD_RUN_MAP_KEY = "ui_compare_add_run_map"
COMPARE_DASHBOARD_SYNC_KEY = "ui_compare_dashboard_sync"
COMPARE_CHECKBOX_PREFIX = "ui_compare_select_"
FLASH_MESSAGE_KEY = "ui_flash_message"
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
DATASET_BASE_KEY = "ui_dataset_base"
DATASET_ROLE_BASE_KEY = "ui_dataset_roles_initialized_for"
TEMPLATE_OVERRIDES_KEY = "ui_template_overrides_text"
DASHBOARD_NAME_FILTER_KEY = "ui_dashboard_filter_name"
DASHBOARD_METHOD_FILTER_KEY = "ui_dashboard_filter_method"
DASHBOARD_DATASET_FILTER_KEY = "ui_dashboard_filter_dataset"
DASHBOARD_STATUS_FILTER_KEY = "ui_dashboard_filter_status"
SIDEBAR_BOOTSTRAP_KEY = "ui_sidebar_bootstrap_done"
MLFLOW_TARGET_KEY = "ui_mlflow_target"
MLFLOW_TARGET_APPLIED_KEY = "ui_mlflow_target_applied"
PENDING_BROWSER_OPEN_KEY = "ui_pending_browser_open_url"
PENDING_BROWSER_OPEN_NONCE_KEY = "ui_pending_browser_open_nonce"
EXAMPLE_FLASH_KEY = "ui_example_flash"
EXAMPLE_SELECTED_KEY = "ui_example_selected"
COMPARE_METRIC_CONTEXT_KEY = "ui_compare_metric_context_id"
PENDING_SCROLL_TOP_KEY = "ui_pending_scroll_top"

GENERAL_UI_KEYS = {
    VIEW_KEY,
    SELECTED_RUN_KEY,
    COMPARE_RUN_IDS_KEY,
    COMPARE_PICKER_KEY,
    COMPARE_ADD_RUN_KEY,
    COMPARE_ADD_RUN_MAP_KEY,
    COMPARE_DASHBOARD_SYNC_KEY,
    FLASH_MESSAGE_KEY,
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
    MLFLOW_TARGET_KEY,
    MLFLOW_TARGET_APPLIED_KEY,
    PENDING_BROWSER_OPEN_KEY,
    PENDING_BROWSER_OPEN_NONCE_KEY,
    EXAMPLE_FLASH_KEY,
    EXAMPLE_SELECTED_KEY,
    COMPARE_METRIC_CONTEXT_KEY,
    PENDING_SCROLL_TOP_KEY,
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
FEDERATION_BASE_PATHS = [
    "federated_params.amount_of_clients",
    "federated_params.client_subset_size",
    "federated_params.communication_rounds",
    "federated_params.local_epochs",
]
DATA_LOADING_BASE_PATHS = [
    "training_params.batch_size",
    "training_params.num_workers",
    "federated_params.client_train_val_prop",
]
OTHER_BASE_PATHS = [
    "federated_params.print_client_metrics",
    "federated_params.server_saving_metrics",
    "federated_params.server_saving_agg",
]
NO_ATTACK_BASE_PARAMS = {
    "federated_params.clients_attack_types": "no_attack",
    "federated_params.prop_attack_clients": 0.0,
    "federated_params.attack_scheme": "no_attack",
    "federated_params.prop_attack_rounds": 0.0,
}
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
    "method",
    "selector",
    "dataset",
    "attacks",
    "setup",
    "launch",
]
CREATE_STEP_LABELS = {
    "template": "1. Template",
    "method": "2. FL Method",
    "selector": "3. Client Selection",
    "dataset": "4. Dataset",
    "attacks": "5. Attacks",
    "setup": "6. Experiment Setup",
    "launch": "7. Review & Launch",
}

BRAND_FED_COLOR = "#111827"
BRAND_XPLORE_COLOR = "#111827"
APP_BACKGROUND_COLOR = "#F7F8FA"
APP_TEXT_COLOR = "#111827"


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
        return
    st.experimental_rerun()


def queue_browser_open(url: str) -> None:
    st.session_state[PENDING_BROWSER_OPEN_KEY] = str(url or "").strip()
    st.session_state[PENDING_BROWSER_OPEN_NONCE_KEY] = time.time_ns()


def queue_scroll_top() -> None:
    """Scroll after a navigation transition, never on fragment refreshes."""

    st.session_state[PENDING_SCROLL_TOP_KEY] = time.time_ns()


def render_pending_scroll_top() -> None:
    nonce = int(st.session_state.get(PENDING_SCROLL_TOP_KEY, 0) or 0)
    if not nonce:
        return
    st.session_state[PENDING_SCROLL_TOP_KEY] = 0
    components.html(
        f"""
        <div style="display:none">{nonce}</div>
        <script>
        window.parent.scrollTo({{top: 0, left: 0, behavior: "auto"}});
        </script>
        """,
        height=0,
        width=0,
    )


def render_pending_browser_open() -> None:
    url = str(st.session_state.get(PENDING_BROWSER_OPEN_KEY, "") or "").strip()
    nonce = int(st.session_state.get(PENDING_BROWSER_OPEN_NONCE_KEY, 0) or 0)
    if not url:
        return
    st.session_state[PENDING_BROWSER_OPEN_KEY] = ""
    st.session_state[PENDING_BROWSER_OPEN_NONCE_KEY] = 0
    components.html(
        f"""
        <div style="display:none">{nonce}</div>
        <script>
        const targetUrl = {json.dumps(url)};
        const targetNonce = {nonce};
        window.parent.open(targetUrl, "_blank", "noopener,noreferrer");
        window.__fedxploreLastOpenedMlflowNonce = targetNonce;
        </script>
        """,
        height=0,
        width=0,
    )


def install_keyboard_guard() -> None:
    components.html(
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
        height=0,
        width=0,
    )


def install_history_navigation_sync() -> None:
    """Reload Streamlit after browser Back/Forward changes query parameters.

    Streamlit updates the address bar for our view routing, but a browser
    history navigation does not always trigger a script rerun by itself.
    """

    components.html(
        """
        <script>
        (function () {
          const appWindow = window.parent;
          if (appWindow.__fedxploreHistorySyncInstalled) {
            return;
          }
          appWindow.__fedxploreHistorySyncInstalled = true;
          appWindow.addEventListener("popstate", function () {
            appWindow.setTimeout(function () {
              appWindow.location.reload();
            }, 0);
          });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def brand_markup(*, suffix: str = "", level: int = 2) -> str:
    safe_level = min(max(level, 1), 6)
    suffix_html = f" <span class='fx-brand-suffix'>{suffix}</span>" if suffix else ""
    return (
        f"<h{safe_level} class='fx-brand-title'>"
        f"<span class='fx-brand-fed'>Fed</span>"
        f"<span class='fx-brand-xplore'>Xplore</span>"
        f"{suffix_html}"
        f"</h{safe_level}>"
    )


def install_button_palette_hook() -> None:
    components.html(
        """
        <script>
        (function () {
          const rootDoc = window.parent.document;
          const labelsToClass = {
            "Create Run": "fx-button-primary",
            "Run": "fx-button-primary",
            "Launch experiment": "fx-button-primary",
            "Compare selected": "fx-button-primary",
            "Examples": "fx-button-examples",
            "Launch example": "fx-button-examples",
            "Stop": "fx-button-danger"
          };

          const syncButtonClasses = function () {
            rootDoc.querySelectorAll("button").forEach(function (button) {
              button.classList.remove(
                "fx-button-primary",
                "fx-button-danger",
                "fx-button-examples"
              );
              const label = (button.innerText || button.textContent || "").trim();
              const className = labelsToClass[label];
              if (className) {
                button.classList.add(className);
              }
            });
          };

          syncButtonClasses();
          window.setTimeout(syncButtonClasses, 150);
          window.setTimeout(syncButtonClasses, 500);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def install_sidebar_controller(*, auto_expand: bool) -> None:
    auto_expand_literal = "true" if auto_expand else "false"
    components.html(
        f"""
        <script>
        (function () {{
          const rootDoc = window.parent.document;
          const isVisible = function (node) {{
            if (!node) {{
              return false;
            }}
            const rect = node.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          }};

          const getSidebar = function () {{
            return rootDoc.querySelector('section[data-testid="stSidebar"]');
          }};

          const sidebarButtons = function () {{
            return Array.from(rootDoc.querySelectorAll("button"));
          }};

          const matchesSidebarControl = function (button, pattern) {{
            const text = [
              button.getAttribute("aria-label") || "",
              button.getAttribute("title") || "",
              button.textContent || "",
            ].join(" ");
            return pattern.test(text);
          }};

          const findOpenButton = function () {{
            return (
              sidebarButtons().find((button) => isVisible(button) && matchesSidebarControl(button, /(open|expand)\\s+sidebar/i)) ||
              rootDoc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
              rootDoc.querySelector('[data-testid="collapsedControl"] button') ||
              rootDoc.querySelector('header[data-testid="stHeader"] button') ||
              sidebarButtons().find((button) => {{
                const rect = button.getBoundingClientRect();
                return rect.left < 120 && rect.top < 120 && matchesSidebarControl(button, /(sidebar|navigation|open|expand)/i);
              }})
            );
          }};

          const findCloseButton = function () {{
            const sidebar = getSidebar();
            if (!sidebar) {{
              return null;
            }}
            return (
              sidebar.querySelector('button[aria-label="Close sidebar"]') ||
              sidebar.querySelector('button[aria-label="Collapse sidebar"]') ||
              sidebar.querySelector('button[title="Close sidebar"]') ||
              sidebar.querySelector('button[title="Collapse sidebar"]') ||
              Array.from(sidebar.querySelectorAll("button")).find((button) => isVisible(button) && matchesSidebarControl(button, /(close|collapse)\\s+sidebar/i)) ||
              Array.from(sidebar.querySelectorAll("button")).find((button) => {{
                if (!isVisible(button)) {{
                  return false;
                }}
                const rect = button.getBoundingClientRect();
                return rect.top < 140;
              }}) ||
              null
            );
          }};

          const ensureToggle = function () {{
            let toggleRoot = rootDoc.getElementById("fx-sidebar-toggle-root");
            if (!toggleRoot) {{
              toggleRoot = rootDoc.createElement("div");
              toggleRoot.id = "fx-sidebar-toggle-root";
              toggleRoot.innerHTML = `
                <button id="fx-sidebar-toggle" type="button" title="Toggle navigation" aria-label="Toggle navigation">
                  <span id="fx-sidebar-toggle-icon" class="fx-sidebar-toggle-icon" aria-hidden="true">&gt;&gt;</span>
                </button>
              `;
              rootDoc.body.appendChild(toggleRoot);
            }}
            return toggleRoot;
          }};

          const ensureStyles = function () {{
            if (rootDoc.getElementById("fx-sidebar-toggle-styles")) {{
              return;
            }}
            const style = rootDoc.createElement("style");
            style.id = "fx-sidebar-toggle-styles";
            style.textContent = `
              #fx-sidebar-toggle-root {{
                  position: fixed;
                  top: 1rem;
                  z-index: 1000;
                  display: block;
              }}
              #fx-sidebar-toggle {{
                  display: inline-flex;
                  align-items: center;
                  justify-content: center;
                  min-width: 2.9rem;
                  height: 2.8rem;
                  padding: 0 0.72rem;
                  border: 1px solid rgba(15, 118, 110, 0.24);
                  border-radius: 999px;
                  background: rgba(255, 255, 255, 0.96);
                  color: #344054;
                  font-weight: 700;
                  font-size: 1rem;
                  letter-spacing: -0.08em;
                  box-shadow: 0 8px 20px rgba(16, 24, 40, 0.12);
                  backdrop-filter: blur(12px);
                  cursor: pointer;
                  transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
              }}
              #fx-sidebar-toggle:hover {{
                  transform: translateY(-1px);
                  border-color: rgba(15, 118, 110, 0.42);
                  box-shadow: 0 10px 24px rgba(16, 24, 40, 0.16);
              }}
              .fx-sidebar-toggle-icon {{
                  display: inline-flex;
                  align-items: center;
                  justify-content: center;
                  color: #0F766E;
                  font-weight: 800;
                  line-height: 1;
              }}
              @media (max-width: 900px) {{
                  #fx-sidebar-toggle-root {{
                      top: 0.75rem;
                  }}
                  #fx-sidebar-toggle {{
                      min-width: 2.55rem;
                      height: 2.55rem;
                  }}
              }}
            `;
            rootDoc.head.appendChild(style);
          }};

          const clickButton = function (button) {{
            if (!button) {{
              return false;
            }}
            button.click();
            return true;
          }};

          const openSidebar = function () {{
            return clickButton(findOpenButton());
          }};

          const closeSidebar = function () {{
            return clickButton(findCloseButton());
          }};

          const toggleSidebar = function () {{
            const openButton = findOpenButton();
            if (openButton) {{
              return clickButton(openButton);
            }}
            return closeSidebar();
          }};

          ensureStyles();
          const toggleRoot = ensureToggle();
          const toggleButton = rootDoc.getElementById("fx-sidebar-toggle");
          const toggleIcon = rootDoc.getElementById("fx-sidebar-toggle-icon");

          const syncToggleState = function () {{
            const collapsed = Boolean(findOpenButton());
            const sidebar = getSidebar();
            toggleIcon.textContent = collapsed ? ">>" : "<<";
            toggleButton.setAttribute("title", collapsed ? "Expand navigation" : "Collapse navigation");
            toggleButton.setAttribute("aria-label", collapsed ? "Expand navigation" : "Collapse navigation");
            if (!collapsed && sidebar) {{
              const rect = sidebar.getBoundingClientRect();
              const left = Math.max(12, rect.right - 24);
              toggleRoot.style.left = `${{left}}px`;
            }} else {{
              toggleRoot.style.left = "1rem";
            }}
          }};

          if (!rootDoc.__fedxploreSidebarToggleBound) {{
            toggleButton.addEventListener("click", function () {{
              toggleSidebar();
              window.setTimeout(syncToggleState, 120);
            }});
            rootDoc.__fedxploreSidebarToggleBound = true;
          }}

          if (!rootDoc.__fedxploreSidebarToggleInterval) {{
            rootDoc.__fedxploreSidebarToggleInterval = window.setInterval(syncToggleState, 350);
          }}

          syncToggleState();
          if ({auto_expand_literal} && !rootDoc.__fedxploreSidebarAutoExpanded) {{
            rootDoc.__fedxploreSidebarAutoExpanded = true;
            window.setTimeout(function () {{
              openSidebar();
              window.setTimeout(syncToggleState, 120);
            }}, 180);
          }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def normalize_compare_run_ids(run_ids: Any) -> list[str]:
    """Normalize comparison selection while preserving its visible order."""

    if isinstance(run_ids, str):
        candidates = run_ids.split(",")
    elif isinstance(run_ids, (list, tuple, set)):
        candidates = run_ids
    else:
        candidates = []
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        run_id = str(candidate or "").strip()
        if run_id and run_id not in seen:
            normalized.append(run_id)
            seen.add(run_id)
    return normalized


def compare_checkbox_key(run_id: str) -> str:
    return f"{COMPARE_CHECKBOX_PREFIX}{run_id}"


def set_compare_run_ids(run_ids: Any, *, sync_picker: bool = True) -> list[str]:
    normalized = normalize_compare_run_ids(run_ids)
    st.session_state[COMPARE_RUN_IDS_KEY] = normalized
    if sync_picker:
        st.session_state[COMPARE_PICKER_KEY] = list(normalized)
    st.session_state[COMPARE_DASHBOARD_SYNC_KEY] = None
    return normalized


def clear_compare_selection() -> None:
    """Clear comparison state before returning to the independent run list."""

    set_compare_run_ids([])


def sync_query_params(
    view: str,
    run_id: str | None = None,
    compare_run_ids: list[str] | None = None,
) -> None:
    compare_ids = normalize_compare_run_ids(compare_run_ids)
    if hasattr(st, "query_params"):
        st.query_params.clear()
        st.query_params["view"] = view
        if run_id:
            st.query_params["run_id"] = run_id
        if compare_ids:
            st.query_params["runs"] = ",".join(compare_ids)
        return
    params: dict[str, str] = {"view": view}
    if run_id:
        params["run_id"] = run_id
    if compare_ids:
        params["runs"] = ",".join(compare_ids)
    st.experimental_set_query_params(**params)


def restore_view_from_query_params() -> None:
    if hasattr(st, "query_params"):
        view = st.query_params.get("view", "")
        run_id = st.query_params.get("run_id", "")
        compare_run_ids = st.query_params.get("runs", "")
    else:
        params = st.experimental_get_query_params()
        view = params.get("view", [""])[0]
        run_id = params.get("run_id", [""])[0]
        compare_run_ids = params.get("runs", [""])[0]
    previous_view = str(st.session_state.get(VIEW_KEY, "") or "")
    if view in {VIEW_DASHBOARD, VIEW_CREATE, VIEW_RUN, VIEW_COMPARE, VIEW_EXAMPLES}:
        st.session_state[VIEW_KEY] = view
        if view == VIEW_COMPARE and previous_view != VIEW_COMPARE:
            queue_scroll_top()
    if run_id:
        st.session_state[SELECTED_RUN_KEY] = run_id
    if compare_run_ids:
        set_compare_run_ids(compare_run_ids)


def apply_page_styles() -> None:
    inject_global_styles()


def pick_options(discovered: list[str], fallback: list[str]) -> list[str]:
    return discovered or fallback


def get_option_sets(repo_root: Path) -> dict[str, list[str]]:
    return {
        "dataset": pick_options(
            discover_config_group_options(
                repo_root, "dataset", exclude={"federated_dataset"}
            ),
            ["cifar10", "cifar100", "ptbxl", "tiny_imagenet"],
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
    mlflow_target = infer_mlflow_target(
        mlflow_defaults.get("tracking_uri"),
        remote_tracking_uri=mlflow_defaults.get("remote_tracking_uri"),
    )
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
        COMPARE_RUN_IDS_KEY: [],
        COMPARE_PICKER_KEY: [],
        COMPARE_DASHBOARD_SYNC_KEY: None,
        FLASH_MESSAGE_KEY: "",
        LAST_LINES_KEY: 200,
        TEMPLATE_PICKER_KEY: "",
        LOADED_TEMPLATE_KEY: "",
        CREATE_STEP_KEY: CREATE_STEPS[0],
        EXAMPLE_FLASH_KEY: "",
        EXAMPLE_SELECTED_KEY: "",
        COMPARE_METRIC_CONTEXT_KEY: "",
        PENDING_SCROLL_TOP_KEY: 0,
        PENDING_TEMPLATE_KEY: "",
        PENDING_RESET_KEY: False,
        PENDING_STEP_KEY: "",
        SIDEBAR_BOOTSTRAP_KEY: False,
        "ui_run_name": DEFAULT_RUN_NAME,
        "ui_mlflow_ui_url": mlflow_defaults["ui_url"],
        MLFLOW_TARGET_KEY: mlflow_target,
        MLFLOW_TARGET_APPLIED_KEY: "",
        "ui_raw_overrides": "",
        TEMPLATE_OVERRIDES_KEY: "",
        DEVICE_MODE_KEY: str(main_flat_defaults.get("training_params.device", "cuda")),
        DEVICE_IDS_KEY: list(main_flat_defaults.get("training_params.device_ids", [0])),
        DATASET_BASE_KEY: default_selections.get("train_dataset") or (
            options.get("dataset", [""])[0] if options.get("dataset") else ""
        ),
        DATASET_ROLE_BASE_KEY: "",
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

def navigate_to(
    view: str,
    *,
    run_id: str | None = None,
    compare_run_ids: list[str] | None = None,
) -> None:
    st.session_state[VIEW_KEY] = view
    if run_id is not None:
        st.session_state[SELECTED_RUN_KEY] = run_id
    if compare_run_ids is not None:
        set_compare_run_ids(compare_run_ids)
    if view == VIEW_COMPARE:
        queue_scroll_top()
    active_compare_ids = (
        normalize_compare_run_ids(st.session_state.get(COMPARE_RUN_IDS_KEY, []))
        if view == VIEW_COMPARE
        else []
    )
    sync_query_params(view, run_id, active_compare_ids)


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
    st.session_state[DATASET_BASE_KEY] = str(
        st.session_state.get("ui_train_dataset", "") or ""
    )
    # A template is authoritative: opening the Dataset step must not replace
    # its independently chosen train/test/trust roles.
    st.session_state[DATASET_ROLE_BASE_KEY] = st.session_state[DATASET_BASE_KEY]
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


def render_create_summary() -> None:
    """Persistent research-level context for the Create Run flow."""

    rows = build_experiment_summary(st.session_state)
    with st.container(key="create-summary"):
        st.markdown("<div class='fx-summary-title'>Experiment Summary</div>", unsafe_allow_html=True)
        if not rows:
            st.caption("Choose a template or configure the experiment to see a summary.")
            return
        for label, value in rows:
            st.markdown(
                f"<div class='fx-summary-row'><div class='fx-summary-label'>{label}</div>"
                f"<div class='fx-summary-value'>{value}</div></div>",
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
    on_change=None,
) -> None:
    if none_label is None:
        st.selectbox(
            label,
            options,
            key=key,
            format_func=lambda value: str(value),
            on_change=on_change,
        )
        return
    st.selectbox(
        label,
        options,
        key=key,
        format_func=lambda value: none_label if not value else str(value),
        on_change=on_change,
    )


def reset_no_attack_settings() -> None:
    """Restore a coherent disabled-attack configuration after switching back."""

    if str(st.session_state.get(ATTACK_TYPE_KEY, "no_attack")) != "no_attack":
        return
    for path, value in NO_ATTACK_BASE_PARAMS.items():
        st.session_state[base_widget_key(path)] = value


def resolve_template_key(raw_value: str, templates: dict[str, TemplateSpec]) -> str:
    if raw_value in templates:
        return raw_value
    for key, template in templates.items():
        if template.name == raw_value:
            return key
    return ""


def apply_mlflow_target_preset(repo_root: Path, target: str) -> None:
    mlflow_defaults = get_mlflow_defaults(repo_root)
    tracking_uri_key = component_widget_key("logger", "mlflow", "tracking_uri")
    if target == "remote":
        tracking_uri = (
            mlflow_defaults.get("remote_tracking_uri", "").strip() or mlflow_defaults["tracking_uri"]
        )
        ui_url = normalize_mlflow_ui_url(tracking_uri)
    else:
        tracking_uri = get_local_mlflow_tracking_uri(repo_root)
        ui_url = DEFAULT_LOCAL_MLFLOW_UI_URL

    st.session_state[tracking_uri_key] = tracking_uri
    st.session_state["ui_mlflow_ui_url"] = ui_url
    st.session_state[MLFLOW_TARGET_APPLIED_KEY] = target


def ensure_mlflow_tracking_uri(
    repo_root: Path, target: str, *, force: bool = False
) -> bool:
    """Ensure a selected MLflow logger always has a usable tracking URI.

    A Blank template resets component fields to the YAML ``null`` value while
    retaining the selected MLflow target in general UI state.  This helper is
    deliberately also called during payload collection, so jumping straight
    to Review & Launch cannot produce a ``logger.tracking_uri=null`` run.
    """

    tracking_uri_key = component_widget_key("logger", "mlflow", "tracking_uri")
    current_uri = str(st.session_state.get(tracking_uri_key, "") or "").strip()
    if not force and current_uri.lower() not in {"", "null", "none"}:
        return False
    apply_mlflow_target_preset(repo_root, target)
    return True


def reset_navigation_state() -> None:
    """Discard temporary page choices before a top-level sidebar transition."""

    clear_compare_selection()
    st.session_state[EXAMPLE_SELECTED_KEY] = ""
    st.session_state[EXAMPLE_FLASH_KEY] = ""
    st.session_state[COMPARE_METRIC_CONTEXT_KEY] = ""
    for key in list(st.session_state):
        if key.startswith(COMPARE_CHECKBOX_PREFIX):
            del st.session_state[key]
    for key, value in {
        DASHBOARD_NAME_FILTER_KEY: "",
        DASHBOARD_METHOD_FILTER_KEY: "All",
        DASHBOARD_DATASET_FILTER_KEY: "All",
        DASHBOARD_STATUS_FILTER_KEY: "All",
    }.items():
        st.session_state[key] = value


def render_sidebar(defaults: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown(brand_markup(level=2), unsafe_allow_html=True)
        if st.button("Dashboard", key="sidebar_dashboard", use_container_width=True):
            reset_navigation_state()
            navigate_to(VIEW_DASHBOARD)
            rerun_app()
        if st.button("Create Run", key="sidebar_create_run", use_container_width=True):
            reset_navigation_state()
            reset_form_to_defaults(defaults)
            set_create_step(CREATE_STEPS[0])
            navigate_to(VIEW_CREATE)
            rerun_app()
        if st.button("Examples", key="sidebar_examples", use_container_width=True):
            reset_navigation_state()
            navigate_to(VIEW_EXAMPLES)
            rerun_app()


def examples_catalog_path() -> Path:
    return Path(__file__).with_name("examples.yaml")


def select_example(example_key: str) -> None:
    st.session_state[EXAMPLE_SELECTED_KEY] = example_key


def render_example_card(example: ExampleDefinition, *, selected: bool) -> bool:
    state_suffix = "selected" if selected else "normal"
    with st.container(key=f"example-card-selected-{example.key}" if selected else f"example-card-{state_suffix}-{example.key}"):
        data = example.data
        color = escape(str(data.get("category_color", "#7C3AED")))
        is_interdependency = example.key == "interdependency"
        tag_style = (
            "background:#FFF7ED;color:#B54708;"
            if is_interdependency
            else ""
        )
        tags = "".join(
            f"<span class='fx-example-tag' style='{tag_style}'>{escape(str(tag))}</span>"
            for tag in data.get("tags", [])
        )
        st.markdown(
            (
                f"<div class='fx-example-category' style='color:{color}'>{escape(str(data.get('category', 'EXAMPLE')))}</div>"
                f"<div class='fx-example-title'>{escape(example.title)}</div>"
            ),
            unsafe_allow_html=True,
        )
        st.image(Path(__file__).parent / str(data["poster"]), use_container_width=True)
        st.markdown(
            (
                f"<div class='fx-example-description'>{escape(str(data.get('description', '')))}</div>"
                f"<div class='fx-example-tags'>{tags}</div>"
                f"<div class='fx-example-footer'>{escape(str(data.get('comparison_footer', '')))}</div>"
            ),
            unsafe_allow_html=True,
        )
        return st.button(
            "Selected" if selected else "Choose example",
            key=f"choose_example_{example.key}",
            disabled=selected,
            on_click=select_example,
            args=[example.key],
            use_container_width=True,
        )


def launch_example(repo_root: Path, example: ExampleDefinition) -> None:
    """Launch a fixed curated suite with the deliberately simple defaults."""

    st.session_state[MLFLOW_TARGET_KEY] = "local"
    apply_mlflow_target_preset(repo_root, "local")
    ensure_mlflow_tracking_uri(repo_root, "local")
    tracking_uri = str(
        st.session_state.get(component_widget_key("logger", "mlflow", "tracking_uri"), "") or ""
    ).strip()
    if not tracking_uri:
        st.error("Local MLflow tracking URI could not be configured.")
        return

    base_env, _ = build_subprocess_env(disable_proxy=True, mlflow_tracking_uri=tracking_uri)
    plan = build_example_launch_plan(
        example,
        group_id=build_group_id(example),
        device="cpu",
        gpu_ids=[],
        seed=int(example.data.get("default_seed", 42)),
        tracking_uri=tracking_uri,
        base_env=base_env,
    )
    progress = st.progress(0, text="Starting example runs…")
    with st.status("Starting example runs…", expanded=True) as status_box:
        attempted = 0

        def on_attempt(request, started, error) -> None:
            nonlocal attempted
            attempted += 1
            mark = "✓" if started else "✗"
            status_box.write(f"{mark} {attempted} / {len(plan)}  {request.display_label}")
            progress.progress(attempted / len(plan), text=f"Starting {attempted} / {len(plan)} runs…")

        statuses, errors = launch_example_suite(
            plan,
            start_run,
            repo_root=repo_root,
            mlflow_url=str(st.session_state.get("ui_mlflow_ui_url", "") or normalize_mlflow_ui_url(tracking_uri)),
            on_attempt=on_attempt,
        )
        status_box.update(label="Example launch complete", state="complete")
    if not statuses:
        st.error("No child run could be started. " + " ".join(errors))
        return
    if errors:
        st.session_state[EXAMPLE_FLASH_KEY] = "Some child runs could not start: " + " | ".join(errors)
    clear_compare_selection()
    run_ids = [str(status["run_id"]) for status in statuses]
    set_compare_run_ids(run_ids)
    navigate_to(VIEW_COMPARE, compare_run_ids=run_ids)
    rerun_app()


def render_examples_page(repo_root: Path) -> None:
    st.markdown("<span class='fx-examples-page-marker'></span>", unsafe_allow_html=True)
    examples = load_examples(examples_catalog_path())
    selected_key = str(st.session_state.get(EXAMPLE_SELECTED_KEY, "") or "")
    if selected_key not in examples:
        selected_key = ""
        st.session_state[EXAMPLE_SELECTED_KEY] = ""
    st.markdown(brand_markup(suffix="Examples", level=1), unsafe_allow_html=True)
    st.caption("Choose a curated suite, then launch it on CPU with local MLflow tracking.")
    cards = st.columns(2, gap="large")
    for column, example in zip(cards, ordered_examples(examples)):
        with column:
            render_example_card(example, selected=example.key == selected_key)
    launch_column = st.columns([1.0, 1.25, 1.0])[1]
    with launch_column:
        if st.button(
            "Launch example",
            key="examples_launch",
            disabled=not selected_key,
            use_container_width=True,
        ):
            launch_example(repo_root, examples[selected_key])


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


def saved_override_values(spec: dict[str, Any]) -> dict[str, str]:
    """Read last-wins Hydra values for runs created outside the form UI."""

    values: dict[str, str] = {}
    overrides = spec.get("overrides", [])
    if not isinstance(overrides, list):
        return values
    for raw_override in overrides:
        override = str(raw_override).strip()
        while override.startswith(("+", "~")):
            override = override[1:]
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def extract_run_meta(repo_root: Path, run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["run_dir"])
    spec = read_spec(run_dir)
    payload = spec.get("form_payload", {})
    selections = payload.get("selected_groups", {})
    base_params = payload.get("base_params", {})
    old_form_values = spec.get("form_values", {})
    override_values = saved_override_values(spec)

    dataset_name = first_non_empty(
        selections.get("train_dataset"),
        old_form_values.get("dataset"),
        override_values.get("dataset@train_dataset"),
    )
    method_name = first_non_empty(
        selections.get("federated_method"),
        old_form_values.get("federated_method"),
        override_values.get("federated_method"),
    )
    logger_name = first_non_empty(
        selections.get("logger"),
        old_form_values.get("logger"),
        override_values.get("logger"),
    )
    rounds = first_non_empty(
        base_params.get("federated_params.communication_rounds"),
        old_form_values.get("communication_rounds"),
        override_values.get("federated_params.communication_rounds"),
    )
    clients = first_non_empty(
        base_params.get("federated_params.amount_of_clients"),
        old_form_values.get("amount_of_clients"),
        override_values.get("federated_params.amount_of_clients"),
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


def get_run_mlflow_context(repo_root: Path, run: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    payload = meta.get("payload", {}) or {}
    selected_groups = payload.get("selected_groups", {}) or {}
    component_params = payload.get("component_params", {}) or {}
    logger_name = str(selected_groups.get("logger") or meta.get("logger") or "").strip()
    logger_params = component_params.get("logger", {}) or {}
    tracking_uri = str(logger_params.get("tracking_uri", "") or "").strip()
    spec = meta.get("spec", {}) or {}
    saved_overrides = spec.get("overrides", [])
    if isinstance(saved_overrides, list):
        for raw_override in saved_overrides:
            override = str(raw_override).strip()
            while override.startswith(("+", "~")):
                override = override[1:]
            if "=" not in override:
                continue
            key, value = override.split("=", 1)
            if key.strip() == "logger":
                logger_name = value.strip()
            elif key.strip() == "logger.tracking_uri":
                tracking_uri = value.strip()
    ui_url_hint = str(meta.get("mlflow_url", "") or "").strip()
    target = str(payload.get("mlflow_target", "") or "").strip()
    if not target:
        target = infer_mlflow_target(
            tracking_uri,
            remote_tracking_uri=get_mlflow_defaults(repo_root).get("remote_tracking_uri"),
        )

    status = read_status(Path(run["run_dir"]))
    mlflow_run_id = str(
        status.get("mlflow_run_id", "") or spec.get("mlflow_run_id", "") or ""
    ).strip()
    mlflow_experiment_id = str(
        status.get("mlflow_experiment_id", "")
        or spec.get("mlflow_experiment_id", "")
        or ""
    ).strip()
    mlflow_url = str(
        status.get("mlflow_url", "") or spec.get("mlflow_url", "") or ui_url_hint
    ).strip()

    return {
        "enabled": logger_name == "mlflow",
        "target": target,
        "tracking_uri": tracking_uri,
        "ui_url_hint": ui_url_hint,
        "run_id": mlflow_run_id,
        "experiment_id": mlflow_experiment_id,
        "url": mlflow_url,
        "status": status,
    }


def format_rel_path(repo_root: Path, path_value: str | Path) -> str:
    path = Path(path_value)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _sync_dashboard_compare_selection(runs: list[dict[str, Any]]) -> list[str]:
    known_ids = {str(run["run_id"]) for run in runs}
    selected = [
        run_id
        for run_id in normalize_compare_run_ids(
            st.session_state.get(COMPARE_RUN_IDS_KEY, [])
        )
        if run_id in known_ids
    ]
    if selected != st.session_state.get(COMPARE_RUN_IDS_KEY, []):
        st.session_state[COMPARE_RUN_IDS_KEY] = selected

    marker = tuple(selected)
    missing_checkbox_state = any(
        compare_checkbox_key(str(run["run_id"])) not in st.session_state
        for run in runs
    )
    if (
        st.session_state.get(COMPARE_DASHBOARD_SYNC_KEY) != marker
        or missing_checkbox_state
    ):
        selected_set = set(selected)
        for run in runs:
            run_id = str(run["run_id"])
            st.session_state[compare_checkbox_key(run_id)] = run_id in selected_set
        st.session_state[COMPARE_DASHBOARD_SYNC_KEY] = marker
    return selected


def _update_dashboard_compare_selection(run_id: str) -> None:
    selected = normalize_compare_run_ids(
        st.session_state.get(COMPARE_RUN_IDS_KEY, [])
    )
    checkbox_value = bool(st.session_state.get(compare_checkbox_key(run_id), False))
    if checkbox_value and run_id not in selected:
        selected.append(run_id)
    if not checkbox_value:
        selected = [current_id for current_id in selected if current_id != run_id]
    set_compare_run_ids(selected)


def render_dashboard_page(repo_root: Path, runs: list[dict[str, Any]]) -> None:
    total_runs = len(runs)
    running_runs = sum(1 for run in runs if run.get("status") == "running")
    stopping_runs = sum(1 for run in runs if run.get("status") == "stopping")
    selected_run_ids = _sync_dashboard_compare_selection(runs) if runs else []

    header_cols = st.columns([5.0, 1.1, 1.45, 1.35])
    with header_cols[0]:
        st.markdown(brand_markup(suffix="Runs", level=1), unsafe_allow_html=True)
    with header_cols[1]:
        if selected_run_ids:
            st.markdown(
                f"<div class='fx-selection-count'>{len(selected_run_ids)} selected</div>",
                unsafe_allow_html=True,
            )
    with header_cols[2]:
        if len(selected_run_ids) >= 2:
            if st.button(
                "Compare selected",
                key="dashboard_compare_selected",
                use_container_width=True,
            ):
                navigate_to(VIEW_COMPARE, compare_run_ids=selected_run_ids)
                rerun_app()
    with header_cols[3]:
        if st.button("Create Run", key="dashboard_create_run", use_container_width=True):
            set_create_step(CREATE_STEPS[0])
            navigate_to(VIEW_CREATE)
            rerun_app()

    cards = st.columns(3)
    with cards[0]:
        render_card("Runs", str(total_runs))
    with cards[1]:
        render_card("Running", str(running_runs))
    with cards[2]:
        render_card("Stopping", str(stopping_runs))

    if not runs:
        st.info("No runs yet. Create a run to start building your experiment history.")
        return

    metas = [extract_run_meta(repo_root, run) for run in runs]
    method_options = ["All"] + sorted({meta["method"] for meta in metas if meta["method"] not in {"", "-"}})
    dataset_options = ["All"] + sorted({meta["dataset"] for meta in metas if meta["dataset"] not in {"", "-"}})
    status_options = ["All"] + sorted({meta["status"] for meta in metas if meta["status"]})

    with st.container(key="dashboard-runs-table"):
        with st.expander("Search and filters", expanded=False):
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
        header_cols = st.columns([3.0, 1.2, 1.2, 1.3, 1.0, 0.9, 0.8])
        header_labels = ["Name", "Method", "Dataset", "Created", "Time", "Status", ""]
        for col, label in zip(header_cols, header_labels):
            with col:
                st.markdown(f"<div class='fx-table-header'>{label}</div>", unsafe_allow_html=True)

        st.markdown("<div class='fx-divider'></div>", unsafe_allow_html=True)

        if not filtered_rows:
            st.caption("No runs match the current filters.")
            return

        for index, (run, meta) in enumerate(filtered_rows):
            with st.container(key=f"run-row-{safe_key(str(run['run_id']))}"):
                row_cols = st.columns(
                    [3.0, 1.2, 1.2, 1.3, 1.0, 0.9, 0.8],
                    vertical_alignment="center",
                )
                with row_cols[0]:
                    name_cols = st.columns([0.16, 2.84], vertical_alignment="center")
                    with name_cols[0]:
                        st.checkbox(
                            "Select run",
                            key=compare_checkbox_key(str(run["run_id"])),
                            on_change=_update_dashboard_compare_selection,
                            args=[str(run["run_id"])],
                            label_visibility="collapsed",
                        )
                    with name_cols[1]:
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
                render_value_widget(readable_label(path), widget_key, default_value)


def render_create_card(title: str, description: str, *, key: str):
    """Return a keyed, consistently styled card container for Create Run."""

    container = st.container(key=f"create-card-{key}")
    heading = f"<div class='fx-create-card-title'>{title}</div>" if title else ""
    detail = f"<div class='fx-create-card-description'>{description}</div>" if description else ""
    if heading or detail:
        container.markdown(heading + detail, unsafe_allow_html=True)
    return container


def render_advanced_settings(title: str, render_body) -> None:
    """Use one progressive-disclosure pattern across Create Run cards."""

    with st.expander(f"Advanced {title}", expanded=False):
        render_body()


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


def render_component_section(
    repo_root: Path,
    component_name: str,
    option: str,
    *,
    primary_paths: set[str] | None = None,
    hidden_paths: set[str] | None = None,
    advanced_extra=None,
) -> None:
    flat_defaults = get_component_flat_defaults(repo_root, component_name, option)
    if not flat_defaults:
        return
    # Targets describe Python implementation details.  They are needed by
    # Hydra but must never be exposed as an editable experiment parameter.
    hidden = set(hidden_paths or set()) | {"_target_", "config._target_"}
    flat_defaults = {path: value for path, value in flat_defaults.items() if path not in hidden}
    primary = set(flat_defaults) if primary_paths is None else set(primary_paths)
    visible = {path: value for path, value in flat_defaults.items() if path in primary}
    advanced = {path: value for path, value in flat_defaults.items() if path not in primary}
    if visible:
        render_flat_param_editors(
            "",
            visible,
            widget_key_builder=lambda path: component_widget_key(component_name, option, path),
        )
    if advanced or advanced_extra is not None:
        def render_advanced_body() -> None:
            if advanced:
                render_flat_param_editors(
                    "",
                    advanced,
                    widget_key_builder=lambda path: component_widget_key(component_name, option, path),
                )
            if advanced_extra is not None:
                advanced_extra()

        component_label = COMPONENT_LABELS.get(component_name, readable_option(component_name))
        render_advanced_settings(
            f"{component_label.lower()} settings",
            render_advanced_body,
        )


def render_optimizer_betas(repo_root: Path, optimizer: str) -> None:
    """Expose common optimizer betas as two numeric values, preserving the list."""

    defaults = get_component_flat_defaults(repo_root, "optimizer", optimizer)
    if "betas" not in defaults:
        return
    value_key = component_widget_key("optimizer", optimizer, "betas")
    seed_state_value(value_key, defaults["betas"])
    try:
        values = yaml.safe_load(str(st.session_state[value_key]))
    except yaml.YAMLError:
        values = defaults["betas"]
    if not isinstance(values, list) or len(values) < 2:
        values = defaults["betas"]
    beta_keys = (f"{value_key}__beta1", f"{value_key}__beta2")
    st.session_state.setdefault(beta_keys[0], float(values[0]))
    st.session_state.setdefault(beta_keys[1], float(values[1]))
    beta_columns = st.columns(2)
    with beta_columns[0]:
        st.number_input("β1", key=beta_keys[0], format="%.6g")
    with beta_columns[1]:
        st.number_input("β2", key=beta_keys[1], format="%.6g")
    st.session_state[value_key] = [
        float(st.session_state[beta_keys[0]]),
        float(st.session_state[beta_keys[1]]),
    ]


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


def is_hydra_target_override(override: str) -> bool:
    """Keep implementation targets out of both guided and raw UI input."""

    key = override.split("=", 1)[0].lstrip("+").strip()
    return key == "_target_" or key.endswith("._target_")


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
    if attack_type == "no_attack":
        # Defend against a stale session or a launch that happens without
        # returning to the Attacks step.  ``constant`` with zero rounds is
        # invalid in the training code, whereas the disabled state is valid.
        for path, value in NO_ATTACK_BASE_PARAMS.items():
            base_params[path] = value
    attack_params, attack_errors = collect_component_values(
        repo_root, "attack", attack_type
    ) if attack_type != "no_attack" else ({}, [])

    if selected_groups.get("logger") == "mlflow":
        ensure_mlflow_tracking_uri(
            repo_root,
            str(st.session_state.get(MLFLOW_TARGET_KEY, "local") or "local"),
        )

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
    if any(is_hydra_target_override(override) for override in raw_overrides):
        raw_errors.append("Hydra _target_ overrides are managed by the selected component and cannot be edited here.")

    template_override_text = st.session_state.get(TEMPLATE_OVERRIDES_KEY, "")
    template_errors: list[str] = []
    template_overrides: list[str] = []
    try:
        template_overrides = parse_raw_overrides(template_override_text)
    except ValueError as exc:
        template_errors.append(str(exc))
    if any(is_hydra_target_override(override) for override in template_overrides):
        template_errors.append("Template Hydra _target_ overrides are not supported in the UI.")

    form_payload = {
        "run_name": st.session_state["ui_run_name"].strip() or DEFAULT_RUN_NAME,
        "selected_groups": selected_groups,
        "base_params": base_params,
        "component_params": component_params,
        "attack_type": attack_type,
        "attack_params": attack_params,
        "mlflow_target": str(st.session_state.get(MLFLOW_TARGET_KEY, "remote") or "remote"),
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
    st.markdown("### Start from a template")
    st.caption("Templates apply the same saved form values and Hydra overrides as before.")
    active_template = resolve_template_key(
        str(st.session_state.get(LOADED_TEMPLATE_KEY, "") or ""), templates
    )
    current_step = str(st.session_state.get(CREATE_STEP_KEY, "template") or "template")
    cards = [("", "Blank experiment", "Start from the repository defaults.")]
    cards.extend((key, template.name, template.description or "Saved experiment setup.") for key, template in templates.items())
    for start in range(0, len(cards), 2):
        columns = st.columns(2)
        for column, (template_key, title, description) in zip(columns, cards[start : start + 2]):
            selected = template_key == active_template
            card_key = f"template-card-{'selected-' if selected else ''}{safe_key(template_key or 'blank')}"
            with column, st.container(key=card_key):
                st.markdown(f"<div class='fx-create-card-title'>{title}</div>", unsafe_allow_html=True)
                st.caption(description)
                action = "Selected" if selected else "Use template"
                if st.button(action, key=f"use_template_{safe_key(template_key or 'blank')}", disabled=selected, use_container_width=True):
                    if template_key:
                        queue_template_load(template_key, target_step=current_step)
                    else:
                        queue_template_reset(target_step=current_step)
                    rerun_app()


TAG_COLORS = {
    "Personalization": ("#7C3AED", "#F5F3FF"),
    "Byzantine": ("#DC2626", "#FEF2F2"),
    "Heterogeneity": ("#D97706", "#FFFBEB"),
    "Baseline": ("#64748B", "#F1F5F9"),
}


def render_research_tags(tags: list[str]) -> None:
    if not tags:
        return
    rendered = []
    for tag in tags:
        dot, background = TAG_COLORS.get(str(tag), ("#64748B", "#F1F5F9"))
        rendered.append(
            f"<span class='fx-research-tag' style='--tag-dot:{dot};--tag-bg:{background}'>"
            f"<i></i>{tag}</span>"
        )
    st.markdown("<div class='fx-research-tags'>" + "".join(rendered) + "</div>", unsafe_allow_html=True)


def render_research_reference(metadata: dict[str, Any], *, dataset: bool = False) -> None:
    reference = metadata.get("reference")
    if not isinstance(reference, dict) or not reference.get("url"):
        return
    label = str(reference.get("label") or "Reference")
    action = "View dataset ↗" if dataset else "View paper ↗"
    st.caption(label)
    st.markdown(f"[{action}]({reference['url']})")


def select_catalog_option(state_key: str, option: str) -> None:
    st.session_state[state_key] = option


def select_base_dataset(option: str) -> None:
    """Initialize dataset roles only when a user intentionally changes base data."""

    previous = str(st.session_state.get(DATASET_ROLE_BASE_KEY, "") or "")
    roles = initial_dataset_roles(option, previous)
    st.session_state[DATASET_BASE_KEY] = option
    if roles is None:
        return
    st.session_state["ui_train_dataset"] = roles["train_dataset"]
    st.session_state["ui_test_dataset"] = roles["test_dataset"]
    st.session_state["ui_trust_dataset"] = roles["trust_dataset"]
    st.session_state[DATASET_ROLE_BASE_KEY] = option


def render_research_catalog(
    catalog: dict[str, dict[str, dict[str, Any]]],
    section: str,
    options: list[str],
    *,
    selected: str,
    state_key: str,
    include_tags: bool = False,
    base_dataset: bool = False,
) -> None:
    with st.container(key=f"research-catalog-list-{section}"):
        for option in ordered_options(catalog, section, options):
            metadata = metadata_for(catalog, section, option)
            is_selected = option == selected
            key_suffix = "selected" if is_selected else "muted" if selected else "normal"
            with st.container(key=f"research-catalog-{section}-{safe_key(option)}-{key_suffix}"):
                callback = select_base_dataset if base_dataset else select_catalog_option
                args = (option,) if base_dataset else (state_key, option)
                if section == "federated_method":
                    # Method cards have tags and remain compact, with one
                    # transparent button covering the complete card.
                    st.markdown(
                        f"<div class='fx-research-catalog-name'>{metadata['display_name']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"Select {metadata['display_name']}",
                        key=f"catalog_pick_{section}_{safe_key(option)}",
                        on_click=callback,
                        args=args,
                        use_container_width=True,
                    )
                else:
                    # The remaining catalogs are short: use regular, larger
                    # buttons so their labels always stay centered inside.
                    st.button(
                        metadata["display_name"],
                        key=f"catalog_pick_{section}_{safe_key(option)}",
                        on_click=callback,
                        args=args,
                        use_container_width=True,
                    )
                if include_tags:
                    render_research_tags(list(metadata.get("tags", [])))


def render_research_details(
    metadata: dict[str, Any],
    *,
    heading: str | None = None,
    dataset: bool = False,
) -> None:
    st.markdown(f"### {metadata['display_name']}")
    render_research_tags(list(metadata.get("tags", [])))
    st.write(str(metadata.get("description") or ""))
    render_research_reference(metadata, dataset=dataset)
    paper_reference = metadata.get("paper_reference")
    if dataset and isinstance(paper_reference, dict) and paper_reference.get("url"):
        st.markdown(f"[View dataset paper ↗]({paper_reference['url']})")
    if heading:
        st.divider()
        st.markdown(f"#### {heading}")


def render_research_hint(
    catalog: dict[str, dict[str, dict[str, Any]]],
    section: str,
    option: str,
) -> None:
    """Show concise curated context beside an ordinary component selector."""

    metadata = metadata_for(catalog, section, option)
    if metadata.get("known") and metadata.get("description"):
        st.caption(str(metadata["description"]))


def render_component_parameters_card(
    repo_root: Path,
    component_name: str,
    option: str,
    *,
    title: str,
    primary_paths: set[str] | None = None,
) -> None:
    """Show parameter controls only when the selected local config has them."""

    if not get_component_flat_defaults(repo_root, component_name, option):
        return
    with render_create_card(title, "", key=f"parameters-{component_name}-{safe_key(option)}"):
        render_component_section(repo_root, component_name, option, primary_paths=primary_paths)


def render_component_catalog_page(
    repo_root: Path,
    options: list[str],
    *,
    catalog: dict[str, dict[str, dict[str, Any]]],
    section: str,
    state_key: str,
    title: str,
    subtitle: str,
    component_name: str,
    include_tags: bool = False,
    primary_paths: set[str] | None = None,
) -> None:
    st.markdown(f"### {title}")
    st.caption(subtitle)
    selected = str(st.session_state.get(state_key, "") or "")
    columns = st.columns([1.15, 2.1, 1.0], gap="large")
    with columns[0]:
        st.markdown("#### Catalog")
        render_research_catalog(
            catalog, section, options, selected=selected, state_key=state_key, include_tags=include_tags
        )
    with columns[1]:
        st.markdown("#### &nbsp;", unsafe_allow_html=True)
        metadata = metadata_for(catalog, section, selected)
        with render_create_card("", "", key=f"detail-{section}"):
            render_research_details(metadata)
        requirements = metadata.get("requirements", {})
        if isinstance(requirements, dict) and requirements.get("requires_trust_dataset"):
            st.info("This method requires a server-side trust dataset. Configure it on the Dataset step.")
        render_component_parameters_card(
            repo_root, component_name, selected,
            title=f"{metadata['display_name']} parameters",
            primary_paths=primary_paths,
        )
    with columns[2]:
        render_create_summary()


def render_selector_step(repo_root: Path, options: dict[str, list[str]], catalog: dict[str, dict[str, dict[str, Any]]]) -> None:
    render_component_catalog_page(
        repo_root,
        options["client_selector"],
        catalog=catalog,
        section="client_selector",
        state_key="ui_client_selector",
        title="Client Selection",
        subtitle="Choose how participating clients are selected for each communication round.",
        component_name="client_selector",
    )


def render_dataset_step(repo_root: Path, options: dict[str, list[str]], catalog: dict[str, dict[str, dict[str, Any]]]) -> None:
    st.markdown("### Dataset")
    st.caption("Choose the base dataset, configure its roles, and define the federated client split.")
    selected = str(st.session_state.get(DATASET_BASE_KEY, st.session_state.get("ui_train_dataset", "")) or "")
    columns = st.columns([1.15, 2.1, 1.0], gap="large")
    with columns[0]:
        st.markdown("#### Dataset catalog")
        render_research_catalog(catalog, "dataset", options["dataset"], selected=selected, state_key=DATASET_BASE_KEY, base_dataset=True)
    with columns[1]:
        st.markdown("#### &nbsp;", unsafe_allow_html=True)
        metadata = metadata_for(catalog, "dataset", selected)
        with render_create_card("", "", key="dataset-detail"):
            render_research_details(metadata, dataset=True)
        with render_create_card("Parameters", "", key="dataset-roles"):
            role_cols = st.columns(3)
            with role_cols[0]:
                render_select_input("Train dataset", options["dataset"], key="ui_train_dataset")
            with role_cols[1]:
                render_select_input("Test dataset", options["dataset"], key="ui_test_dataset")
            with role_cols[2]:
                render_select_input("Trust dataset", [""] + options["dataset"], key="ui_trust_dataset", none_label="None")
        method_metadata = metadata_for(catalog, "federated_method", st.session_state.get("ui_federated_method", ""))
        requirements = method_metadata.get("requirements", {})
        if isinstance(requirements, dict) and requirements.get("requires_trust_dataset") and not st.session_state.get("ui_trust_dataset"):
            st.warning("The selected FL method requires a server-side trust dataset.")
        defaults = get_main_flat_defaults(repo_root)
        with render_create_card("Data loading & split", "", key="dataset-loading"):
            render_flat_param_editors("", {path: defaults[path] for path in DATA_LOADING_BASE_PATHS}, widget_key_builder=base_widget_key)
        with render_create_card("Client data distribution", "", key="dataset-distribution"):
            render_select_input("Distribution", options["distribution"], key="ui_distribution")
            render_component_section(repo_root, "distribution", st.session_state["ui_distribution"], primary_paths={"alpha", "n_clusters", "dominant_ratio"})
    with columns[2]:
        render_create_summary()


def render_preaggregator_picker(repo_root: Path, options: dict[str, list[str]], catalog: dict[str, dict[str, dict[str, Any]]]) -> None:
    selected = str(st.session_state.get("ui_preaggregator", "") or "")
    render_research_catalog(
        catalog, "preaggregator", options["preaggregator"],
        selected=selected, state_key="ui_preaggregator",
    )
    metadata = metadata_for(catalog, "preaggregator", selected)
    if selected:
        st.write(str(metadata["description"]))
        render_research_reference(metadata)
    if selected:
        render_component_parameters_card(repo_root, "preaggregator", selected, title="Pre-aggregation parameters")


def render_attacks_catalog_step(repo_root: Path, options: dict[str, list[str]], catalog: dict[str, dict[str, dict[str, Any]]]) -> None:
    st.markdown("### Attacks")
    st.caption("Optionally introduce malicious clients and configure their adversarial behavior.")
    selected = str(st.session_state.get(ATTACK_TYPE_KEY, "no_attack") or "no_attack")
    # The summary has its own outer column so its height cannot push the
    # scenario and pre-aggregation sections far down the page.
    content_column, summary_column = st.columns([3.25, 1.0], gap="large")
    with content_column:
        columns = st.columns([1.15, 2.1], gap="large")
        with columns[0]:
            st.markdown("#### Attack catalog")
            render_research_catalog(
                catalog,
                "attack",
                options["attack_type"],
                selected=selected,
                state_key=ATTACK_TYPE_KEY,
            )
            if selected != "no_attack":
                selected_preaggregator = str(st.session_state.get("ui_preaggregator", "") or "")
                st.markdown("#### Pre-aggregation")
                render_research_catalog(
                    catalog,
                    "preaggregator",
                    options["preaggregator"],
                    selected=selected_preaggregator,
                    state_key="ui_preaggregator",
                )
        with columns[1]:
            st.markdown("#### &nbsp;", unsafe_allow_html=True)
            metadata = metadata_for(catalog, "attack", selected)
            with render_create_card("", "", key="attack-detail"):
                render_research_details(metadata)
            if selected == "no_attack":
                reset_no_attack_settings()
            else:
                # A pre-aggregation detail card is meaningful only after a
                # concrete choice; no "None" information card is shown.
                selected_preaggregator = str(st.session_state.get("ui_preaggregator", "") or "")
                if selected_preaggregator:
                    preaggregation_metadata = metadata_for(
                        catalog,
                        "preaggregator",
                        selected_preaggregator,
                    )
                    with render_create_card("", "", key="preaggregation-detail"):
                        render_research_details(preaggregation_metadata)

                # Keep all editable attack-related values in one predictable
                # block: scenario, attack-specific values, then optional
                # pre-aggregation values.
                defaults = get_main_flat_defaults(repo_root)
                with render_create_card("Parameters", "", key="attack-parameters"):
                    st.markdown("#### Attack scenario")
                    render_flat_param_editors(
                        "",
                        {
                            path: defaults[path]
                            for path in [
                                "federated_params.prop_attack_clients",
                                "federated_params.prop_attack_rounds",
                            ]
                        },
                        widget_key_builder=base_widget_key,
                        columns=1,
                    )
                    render_select_input(
                        "Attack schedule",
                        ["constant", "random_rounds", "random_clients", "random_rounds_random_clients"],
                        key=base_widget_key("federated_params.attack_scheme"),
                    )
                    if get_component_flat_defaults(repo_root, "attack", selected):
                        st.divider()
                        st.markdown("#### Attack parameters")
                        render_component_section(
                            repo_root,
                            "attack",
                            selected,
                            primary_paths=set(),
                        )
                    if selected_preaggregator and get_component_flat_defaults(
                        repo_root,
                        "preaggregator",
                        selected_preaggregator,
                    ):
                        st.divider()
                        st.markdown("#### Pre-aggregation parameters")
                        render_component_section(
                            repo_root,
                            "preaggregator",
                            selected_preaggregator,
                        )
    with summary_column:
        render_create_summary()


def render_run_setup_step(repo_root: Path) -> None:
    st.markdown("### Experiment")
    st.caption("Name the experiment and set its reproducibility seed.")
    main_flat_defaults = get_main_flat_defaults(repo_root)
    with render_create_card("Experiment identity", "A clear name makes the run easy to find later.", key="experiment"):
        st.text_input("Run name", key="ui_run_name")
        render_value_widget(
            "Random seed",
            base_widget_key("random_state"),
            main_flat_defaults.get("random_state", 42),
        )
    with render_create_card("Output", "The primary log path is assigned when the run starts.", key="output"):
        log_path = preview_stdout_path(repo_root, st.session_state["ui_run_name"])
        st.text_input("Primary log file", value=format_rel_path(repo_root, log_path), disabled=True)
        st.caption("The UI adds a unique run-ID suffix when it starts the run.")


def render_data_setup_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Data & Clients")
    dataset_card = render_create_card("Dataset", "Select datasets, model and trainer for this experiment.", key="dataset")
    with dataset_card:
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
        model_cols = st.columns(2)
        with model_cols[0]:
            render_select_input("Model", options["model"], key="ui_model")
            render_component_section(repo_root, "model", st.session_state["ui_model"], primary_paths=set())
        with model_cols[1]:
            render_select_input("Model trainer", options["model_trainer"], key="ui_model_trainer")
            render_component_section(repo_root, "model_trainer", st.session_state["ui_model_trainer"], primary_paths=set())

    main_flat_defaults = get_main_flat_defaults(repo_root)
    with render_create_card("Federation", "Control client participation and local training cadence.", key="federation"):
        render_flat_param_editors(
            "",
            {path: main_flat_defaults[path] for path in SETUP_BASE_PATHS},
            widget_key_builder=base_widget_key,
        )
    with render_create_card("Data distribution", "Choose how training data is partitioned across clients.", key="distribution"):
        render_select_input("Distribution", options["distribution"], key="ui_distribution")
        render_component_section(
            repo_root, "distribution", st.session_state["ui_distribution"], primary_paths={"alpha", "n_clusters", "dominant_ratio"}
        )


def render_method_step(
    repo_root: Path,
    options: dict[str, list[str]],
    catalog: dict[str, dict[str, dict[str, Any]]],
) -> None:
    render_component_catalog_page(
        repo_root,
        options["federated_method"],
        catalog=catalog,
        section="federated_method",
        state_key="ui_federated_method",
        title="FL Method",
        subtitle="Choose the federated learning method that defines the core learning and aggregation behavior of the experiment.",
        component_name="federated_method",
        include_tags=True,
    )


def render_logging_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    with render_create_card("Experiment tracking", "Record parameters and metrics for reproducible research.", key="tracking"):
        render_select_input("Logger", options["logger"], key="ui_logger")
        if st.session_state["ui_logger"] == "mlflow":
            st.radio(
                "MLflow target",
                options=["remote", "local"],
                key=MLFLOW_TARGET_KEY,
                horizontal=True,
                format_func=lambda value: "Remote" if value == "remote" else "Local",
            )
            selected_target = str(st.session_state.get(MLFLOW_TARGET_KEY, "remote") or "remote")
            applied_target = str(st.session_state.get(MLFLOW_TARGET_APPLIED_KEY, "") or "")
            if ensure_mlflow_tracking_uri(
                repo_root,
                selected_target,
                force=selected_target != applied_target,
            ):
                rerun_app()
        render_component_section(
            repo_root, "logger", st.session_state["ui_logger"], primary_paths={"experiment_name"}
        )
        if st.session_state["ui_logger"] == "mlflow":
            tracking_uri = st.session_state.get(
                component_widget_key("logger", "mlflow", "tracking_uri"),
                "",
            )
            if not st.session_state.get("ui_mlflow_ui_url"):
                if st.session_state.get(MLFLOW_TARGET_KEY) == "local":
                    st.session_state["ui_mlflow_ui_url"] = DEFAULT_LOCAL_MLFLOW_UI_URL
                elif tracking_uri:
                    st.session_state["ui_mlflow_ui_url"] = normalize_mlflow_ui_url(str(tracking_uri))
            with st.expander("Advanced tracking settings", expanded=False):
                if st.session_state.get(MLFLOW_TARGET_KEY) == "local":
                    st.caption("Local MLflow store: " + format_rel_path(repo_root, get_local_mlflow_tracking_uri(repo_root)))
                st.text_input("MLflow UI", key="ui_mlflow_ui_url")

    main_flat_defaults = get_main_flat_defaults(repo_root)
    with render_create_card("Metrics", "Choose what the server retains during federated training.", key="metrics"):
        render_flat_param_editors(
            "", {path: main_flat_defaults[path] for path in OTHER_BASE_PATHS}, widget_key_builder=base_widget_key,
        )


def render_training_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Training")
    st.caption("Configure local optimization and the training objective.")
    base_cols = st.columns(2)
    with base_cols[0], render_create_card("Optimizer", "Local optimizer and its core hyperparameters.", key="optimizer"):
        render_select_input("Optimizer", options["optimizer"], key="ui_optimizer")
        render_component_section(
            repo_root,
            "optimizer",
            st.session_state["ui_optimizer"],
            primary_paths={"lr", "weight_decay", "momentum"},
            hidden_paths={"betas"},
            advanced_extra=lambda: render_optimizer_betas(repo_root, st.session_state["ui_optimizer"]),
        )
    with base_cols[1], render_create_card("Loss", "Training objective used for the selected model.", key="loss"):
        render_select_input("Loss", options["loss"], key="ui_loss")
        render_component_section(
            repo_root,
            "loss",
            st.session_state["ui_loss"],
            primary_paths={"config.reduction", "config.label_smoothing"},
        )


def render_attacks_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    st.markdown("### Attacks & Robustness")
    with render_create_card("Adversarial behavior", "Enable an attack only when studying robustness.", key="attack"):
        render_select_input(
            "Attack type",
            options["attack_type"],
            key=ATTACK_TYPE_KEY,
            on_change=reset_no_attack_settings,
        )
        if str(st.session_state.get(ATTACK_TYPE_KEY, "no_attack")) == "no_attack":
            st.info("No adversarial behavior configured. Select an attack to configure malicious clients.")
            return
        main_flat_defaults = get_main_flat_defaults(repo_root)
        render_select_input(
            "Attack schedule",
            ["constant", "random_rounds", "random_clients", "random_rounds_random_clients"],
            key=base_widget_key("federated_params.attack_scheme"),
        )
        render_flat_param_editors(
            "",
            {path: main_flat_defaults[path] for path in ["federated_params.prop_attack_clients", "federated_params.prop_attack_rounds"]},
            widget_key_builder=base_widget_key,
        )
        render_component_section(
            repo_root,
            "attack",
            str(st.session_state[ATTACK_TYPE_KEY]),
            primary_paths=set(),
        )


def render_technical_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    gpu_rows = query_gpus()
    gpu_options = [int(row["index"]) for row in gpu_rows if isinstance(row.get("index"), int)]
    if DEVICE_MODE_KEY not in st.session_state:
        st.session_state[DEVICE_MODE_KEY] = "cuda"
    if DEVICE_IDS_KEY not in st.session_state:
        st.session_state[DEVICE_IDS_KEY] = gpu_options[:1] if gpu_options else []
    current_device_ids = normalize_device_selection(st.session_state.get(DEVICE_IDS_KEY, []))
    st.session_state[DEVICE_IDS_KEY] = [device_id for device_id in current_device_ids if device_id in gpu_options]
    technical_cols = st.columns([1.4, 1.1, 1.7])
    with technical_cols[0], render_create_card("Runtime", "Execution manager and batch scheduling.", key="runtime"):
        render_select_input("Manager", options["manager"], key="ui_manager")
        render_component_section(repo_root, "manager", st.session_state["ui_manager"])
    with technical_cols[1], render_create_card("Device & compute", "Choose CPU or visible CUDA devices.", key="compute"):
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
                "GPU devices",
                options=gpu_options,
                key=DEVICE_IDS_KEY,
                format_func=lambda value: f"GPU {value}",
            )
        else:
            st.session_state[DEVICE_IDS_KEY] = []
    with technical_cols[2]:
        render_gpu_monitor()


def render_experiment_setup_step(repo_root: Path, options: dict[str, list[str]]) -> None:
    """Consolidate all non-research component controls without dropping fields."""

    st.markdown("### Experiment Setup")
    st.caption("Configure federation scale, training, tracking and runtime resources.")
    catalog = load_research_catalog(Path(__file__).with_name("research_catalog.yaml"))
    main_column, summary_column = st.columns([3.0, 1.0], gap="large")
    with main_column:
        defaults = get_main_flat_defaults(repo_root)
        with render_create_card("Federation", "Configure the scale and schedule of federated training.", key="setup-federation"):
            render_flat_param_editors("", {path: defaults[path] for path in FEDERATION_BASE_PATHS}, widget_key_builder=base_widget_key)
        with render_create_card("Training", "Model, trainer, optimizer and objective.", key="setup-training"):
            model_columns = st.columns(2)
            with model_columns[0]:
                render_select_input("Model", options["model"], key="ui_model")
                render_research_hint(catalog, "model", st.session_state["ui_model"])
                render_component_section(repo_root, "model", st.session_state["ui_model"], primary_paths=set())
            with model_columns[1]:
                render_select_input("Model trainer", options["model_trainer"], key="ui_model_trainer")
                render_component_section(repo_root, "model_trainer", st.session_state["ui_model_trainer"], primary_paths=set())
            st.divider()
            st.markdown("#### Optimizer")
            render_select_input("Optimizer", options["optimizer"], key="ui_optimizer")
            render_component_section(repo_root, "optimizer", st.session_state["ui_optimizer"], primary_paths={"lr", "weight_decay", "momentum"}, hidden_paths={"betas"}, advanced_extra=lambda: render_optimizer_betas(repo_root, st.session_state["ui_optimizer"]))
            st.divider()
            st.markdown("#### Loss")
            render_select_input("Loss", options["loss"], key="ui_loss")
            render_component_section(repo_root, "loss", st.session_state["ui_loss"], primary_paths={"config.reduction", "config.label_smoothing"})
        with render_create_card("Tracking & Metrics", "Experiment logging and saved server metrics.", key="setup-tracking"):
            render_logging_step(repo_root, options)
        with render_create_card("Runtime & Resources", "Execution manager, batching and devices.", key="setup-runtime"):
            render_technical_step(repo_root, options)
    with summary_column:
        render_create_summary()


def render_launch_step(repo_root: Path) -> None:
    st.markdown("### Review & Launch")
    st.caption("Review the resolved experiment configuration before starting the training process.")
    with render_create_card("Experiment review", "High-signal settings that will be used for this run.", key="review"):
        review_rows = build_experiment_summary(st.session_state)
        review_cols = st.columns(2)
        for index, (label, value) in enumerate(review_rows):
            with review_cols[index % 2]:
                st.markdown(f"**{label}:** {value}")

    main_defaults = get_main_flat_defaults(repo_root)
    identity_cols = st.columns(2)
    with identity_cols[0]:
        st.text_input("Run name", key="ui_run_name")
    with identity_cols[1]:
        render_value_widget("Random seed", base_widget_key("random_state"), main_defaults.get("random_state", 42))

    (
        form_payload,
        raw_overrides,
        errors,
        raw_override_text,
        template_overrides,
        template_override_text,
    ) = collect_form_payload(repo_root)
    catalog = load_research_catalog(Path(__file__).with_name("research_catalog.yaml"))
    method_metadata = metadata_for(catalog, "federated_method", form_payload["selected_groups"].get("federated_method", ""))
    requirements = method_metadata.get("requirements", {})
    check_errors, check_warnings = validate_experiment_state(
        st.session_state,
        requires_trust_dataset=isinstance(requirements, dict) and bool(requirements.get("requires_trust_dataset")),
    )
    st.markdown("#### Configuration checks")
    for message in check_errors:
        st.error(message)
    for message in check_warnings:
        st.warning(message)
    if not errors and not check_errors and not check_warnings:
        st.success("Ready to launch")

    if errors or check_errors:
        for message in errors:
            st.error(message)
        run_disabled = True
        overrides: list[str] = []
    else:
        structured_and_user_overrides = build_overrides(form_payload, raw_overrides)
        overrides = [*template_overrides, *structured_and_user_overrides]
        duplicates = find_duplicate_override_keys(overrides)
        cmd = build_command(repo_root, overrides)
        manual_command = format_manual_shell_command(cmd, preview_stdout_path(repo_root, form_payload["run_name"]))
        resolved = {
            "base": unflatten_mapping(form_payload["base_params"]),
            "components": {
                name: unflatten_mapping(values)
                for name, values in form_payload["component_params"].items()
                if values
            },
            "attack": form_payload.get("attack_params", {}),
        }
        with st.expander("Resolved configuration", expanded=False):
            st.code(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=False), language="yaml")
        with st.expander("Launch command", expanded=False):
            st.code(manual_command, language="bash")
            st.caption("UI-launched runs use a unique log filename after their run ID is created.")
        if duplicates:
            st.caption("Duplicate override keys: " + ", ".join(duplicates))
        run_disabled = False

    nav_cols = st.columns([1.1, 2.2, 3.7])
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
            "Launch experiment",
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

    # A saved template already supplies every launch setting.  Its next step
    # is the review screen; users can still open any preceding step in the
    # stepper if they want to alter a value.
    loaded_template = str(st.session_state.get(LOADED_TEMPLATE_KEY, "") or "")
    next_step = (
        "launch"
        if current_step == "template" and loaded_template
        else CREATE_STEPS[min(len(CREATE_STEPS) - 1, current_index + 1)]
    )

    nav_cols = st.columns([1.1, 2.5, 3.4])
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
            "Continue without attack" if current_step == "attacks" and str(st.session_state.get(ATTACK_TYPE_KEY, "no_attack")) == "no_attack" else "Next",
            key=f"create_next_{current_step}",
            disabled=current_index == len(CREATE_STEPS) - 1,
            on_click=set_create_step,
            args=[next_step],
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
    catalog = load_research_catalog(Path(__file__).with_name("research_catalog.yaml"))
    if current_step == "template":
        render_template_section(defaults, templates)
    elif current_step == "method":
        render_method_step(repo_root, options, catalog)
    elif current_step == "selector":
        render_selector_step(repo_root, options, catalog)
    elif current_step == "dataset":
        render_dataset_step(repo_root, options, catalog)
    elif current_step == "attacks":
        render_attacks_catalog_step(repo_root, options, catalog)
    elif current_step == "setup":
        render_experiment_setup_step(repo_root, options)
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


def format_metric_value(value: float) -> str:
    return f"{value:.8g}"


def format_artifact_size(size: int | None) -> str:
    if size is None:
        return "size unavailable"
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return (
                f"{value:.0f} {unit}"
                if unit == "B"
                else f"{value:.1f} {unit}"
            )
        value /= 1024


def artifact_mime_type(artifact: ArtifactMetadata) -> str:
    suffix = Path(artifact.path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".csv": "text/csv",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".json": "application/json",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def load_artifact_for_viewer(
    mlflow_run_id: str,
    tracking_uri: str,
    artifact: ArtifactMetadata,
    *,
    key_prefix: str,
) -> ArtifactDownload:
    cache_key = f"{key_prefix}_artifact_download"
    signature = (mlflow_run_id, artifact.path, artifact.size)
    cached = st.session_state.get(cache_key)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return ArtifactDownload(artifact, data=cached["data"])

    download = download_artifact(mlflow_run_id, tracking_uri, artifact)
    if download.has_data:
        st.session_state[cache_key] = {
            "signature": signature,
            "data": download.data,
        }
    return download


def render_artifact_viewer(
    mlflow_run_id: str,
    tracking_uri: str,
    *,
    key_prefix: str,
) -> None:
    result = list_run_artifacts(mlflow_run_id, tracking_uri)
    if result.error:
        st.warning(result.error)
    if not result.artifacts:
        if not result.error:
            st.info("No artifacts have been logged yet.")
        return

    rows = [
        {"Artifact": artifact.path, "Size": format_artifact_size(artifact.size)}
        for artifact in result.artifacts
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    artifacts_by_path = {artifact.path: artifact for artifact in result.artifacts}
    selector_key = f"{key_prefix}_artifact"
    if st.session_state.get(selector_key, "") not in ["", *artifacts_by_path]:
        st.session_state[selector_key] = ""
    selected_path = st.selectbox(
        "Artifact",
        options=[""] + list(artifacts_by_path),
        key=selector_key,
        format_func=lambda path: (
            "Select an artifact to preview…"
            if not path
            else f"{path} · {format_artifact_size(artifacts_by_path[path].size)}"
        ),
    )
    if not selected_path:
        st.caption("Artifacts are downloaded only after selection.")
        return

    artifact = artifacts_by_path[selected_path]
    download = load_artifact_for_viewer(
        mlflow_run_id,
        tracking_uri,
        artifact,
        key_prefix=key_prefix,
    )
    if download.error or download.data is None:
        st.warning(download.error or "The selected artifact is unavailable.")
        return

    st.download_button(
        "Download selected artifact",
        data=download.data,
        file_name=artifact.name,
        mime=artifact_mime_type(artifact),
        key=f"{key_prefix}_download_{safe_key(artifact.path)}",
    )
    preview = build_artifact_preview(download)
    if preview.error:
        st.info(preview.error)
        return
    if preview.kind is ArtifactKind.IMAGE:
        st.image(download.data, caption=artifact.path)
    elif preview.kind is ArtifactKind.CSV and preview.dataframe is not None:
        st.dataframe(preview.dataframe, use_container_width=True, hide_index=True)
    elif preview.kind is ArtifactKind.TEXT and preview.text is not None:
        language = Path(artifact.path).suffix.lower().lstrip(".") or "text"
        st.code(preview.text, language=language)
    else:
        st.caption("Preview is not available for this file type.")
    if preview.truncated:
        st.caption(
            "Preview truncated; download the artifact to inspect the complete file."
        )


def _render_run_artifacts_view(
    repo_root: Path,
    run: dict[str, Any],
    meta: dict[str, Any],
    auto_refresh: bool,
) -> None:
    fresh_run = next(
        (
            item
            for item in list_runs(repo_root)
            if item.get("run_id") == run.get("run_id")
        ),
        run,
    )
    meta = extract_run_meta(repo_root, fresh_run)
    is_active = meta["status"] in {"running", "stopping"}
    if is_active != auto_refresh:
        rerun_app()
        return
    mlflow_context = get_run_mlflow_context(repo_root, fresh_run, meta)
    if not mlflow_context["enabled"]:
        st.info("No MLflow artifacts were configured for this run.")
        return
    if not mlflow_context["run_id"]:
        if meta["status"] in {"running", "stopping"}:
            st.info(
                "MLflow is still starting; artifacts will appear when its run ID "
                "is available."
            )
        else:
            st.info("MLflow run ID is unavailable for this run. It may be a legacy record.")
        return
    render_artifact_viewer(
        mlflow_context["run_id"],
        mlflow_context["tracking_uri"],
        key_prefix=f"ui_run_{safe_key(str(run['run_id']))}",
    )


def render_run_artifacts_view(
    repo_root: Path,
    run: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    auto_refresh = meta["status"] in {"running", "stopping"}
    run_every = "1s" if auto_refresh else None
    st.fragment(run_every=run_every)(_render_run_artifacts_view)(
        repo_root,
        run,
        meta,
        auto_refresh,
    )


@st.fragment(run_every="1s")
def render_analytics_view(repo_root: Path, run: dict[str, Any], meta: dict[str, Any]) -> None:
    """Render a graceful MLflow-backed single-run research summary."""

    # Fragment reruns do not execute the surrounding run-detail page.  Reload
    # the registry so status, MLflow IDs and newly logged charts appear live.
    fresh_run = next(
        (item for item in list_runs(repo_root) if item.get("run_id") == run.get("run_id")),
        run,
    )
    run = fresh_run
    meta = extract_run_meta(repo_root, run)
    mlflow_context = get_run_mlflow_context(repo_root, run, meta)
    summary_cols = st.columns([5.1, 1.1])
    with summary_cols[0]:
        st.markdown("### Run summary")
    with summary_cols[1]:
        if st.button("Refresh", key="analytics_refresh", use_container_width=True):
            rerun_app()

    render_kv_table(
        [
            ("Run", meta["name"]),
            ("Run id", run["run_id"]),
            ("Status", meta["status"]),
            ("Started", meta["started_at"] or meta["created_at"] or "N/A"),
            ("Finished", meta["finished_at"] or ("Running" if meta["status"] == "running" else "N/A")),
            ("Duration", meta["duration"]),
            ("MLflow run ID", mlflow_context["run_id"] or "N/A"),
        ]
    )

    if not mlflow_context["enabled"]:
        st.info("No MLflow metrics were configured for this run.")
        return
    if not mlflow_context["run_id"]:
        if meta["status"] in {"running", "stopping"}:
            st.info("MLflow is still starting; refresh this tab when the run has logged its ID.")
        else:
            st.info("MLflow run ID is unavailable for this run. It may be a legacy record.")
        return

    result = load_metric_histories(
        mlflow_context["run_id"],
        mlflow_context["tracking_uri"],
    )
    if result.error:
        st.warning(result.error)
    if result.metric_errors:
        st.caption(
            "Some metric histories could not be read: "
            + ", ".join(sorted(result.metric_errors))
        )
    if not result.points:
        if not result.error:
            st.info("No metrics have been logged yet.")
        return

    metrics = available_metric_names(result.points)
    final_metrics = load_final_metrics(result.points)
    if final_metrics and meta["status"] not in {"running", "stopping"}:
        st.markdown("### Final metrics")
        headline_metrics = sorted(
            final_metrics,
            key=lambda item: (
                not any(token in item.metric.lower() for token in ("accuracy", "acc", "loss")),
                item.metric.lower(),
            ),
        )[:3]
        headline_cols = st.columns(len(headline_metrics))
        for column, item in zip(headline_cols, headline_metrics):
            with column:
                render_card(item.metric, format_metric_value(item.value))
        render_final_metrics_table(
            [
                {"Metric": item.metric, "Value": format_metric_value(item.value)}
                for item in final_metrics
            ],
            key="final-metrics",
        )

    st.markdown("### Metric history")
    selector_key = f"analytics_metrics_{safe_key(str(run['run_id']))}"
    if selector_key not in st.session_state:
        st.session_state[selector_key] = metrics[: min(3, len(metrics))]
    selected_metrics = st.multiselect(
        "Metrics",
        options=metrics,
        key=selector_key,
        help="Metric names are discovered directly from the saved MLflow run.",
    )
    if not selected_metrics:
        st.caption("Select one or more metrics to show their history.")
    chart_columns = st.columns(2)
    for index, metric in enumerate(selected_metrics):
        chart_frame = metric_points_frame(
            [point for point in result.points if point.metric == metric],
            run_labels={mlflow_context["run_id"]: meta["name"]},
        )
        if chart_frame.empty or chart_frame["x"].isna().all():
            st.info(f"{metric}: no usable step or timestamp was recorded.")
            continue
        with chart_columns[index % 2]:
            render_metric_chart_card(metric, chart_frame)


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


def render_provenance_view(run: dict[str, Any]) -> None:
    run_dir = Path(run["run_dir"])
    provenance = load_provenance(run_dir)
    if provenance is None:
        st.info("Provenance was not captured for this run.")
        return

    git = provenance.get("git", {}) if isinstance(provenance, dict) else {}
    if not isinstance(git, dict):
        st.warning("Saved provenance has an unexpected format.")
        return

    dirty_value = git.get("dirty")
    dirty_label = "Dirty" if dirty_value is True else "Clean" if dirty_value is False else "N/A"
    st.markdown("### Git")
    render_kv_table(
        [
            ("Captured", provenance.get("captured_at") or "N/A"),
            ("Git", "Available" if git.get("available") else "Unavailable"),
            ("Repository root", git.get("repo_root") or "N/A"),
            ("Git commit", git.get("commit") or "N/A"),
            ("Branch", git.get("branch") or ("Detached HEAD" if git.get("detached_head") else "N/A")),
            ("Git describe", git.get("describe") or "N/A"),
            ("Remote", git.get("remote_origin") or "N/A"),
            ("Working tree", dirty_label),
        ]
    )
    if git.get("error"):
        st.warning(str(git["error"]))

    file_sections = [
        ("Modified files", "modified_files"),
        ("Staged files", "staged_files"),
        ("Unstaged files", "unstaged_files"),
        ("Untracked files", "untracked_files"),
    ]
    for title, field in file_sections:
        files = git.get(field, [])
        with st.expander(title, expanded=False):
            if isinstance(files, list) and files:
                st.code("\n".join(str(path) for path in files), language="text")
            else:
                st.caption("None")

    if dirty_value is not True:
        return

    patch_files = provenance.get("patch_files", {})
    if not isinstance(patch_files, dict):
        return
    patch_sections = [
        ("Combined diff", "combined"),
        ("Staged diff", "staged"),
        ("Unstaged diff", "unstaged"),
    ]
    max_preview_chars = 12_000
    for title, patch_key in patch_sections:
        file_name = patch_files.get(patch_key)
        if not file_name:
            continue
        patch_path = run_dir / str(file_name)
        if not patch_path.is_file():
            continue
        try:
            patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with st.expander(title, expanded=False):
            if not patch_text:
                st.caption("No changes.")
                continue
            preview = patch_text[:max_preview_chars]
            st.code(preview, language="diff")
            if len(preview) < len(patch_text):
                st.caption("Preview truncated. The complete patch is available in Files.")


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

    for label, file_path, language in [
        ("Command", command_path, "bash"),
        ("Specification", spec_path, "yaml"),
        ("Status", status_path, "json"),
    ]:
        st.markdown(f"#### {label}")
        if file_path.is_file():
            st.code(file_path.read_text(encoding="utf-8", errors="replace"), language=language)
        else:
            st.caption(f"Not available: {file_path.name}")

    provenance = load_provenance(run_dir)
    patch_files = provenance.get("patch_files", {}) if isinstance(provenance, dict) else {}
    git = provenance.get("git", {}) if isinstance(provenance, dict) else {}
    if isinstance(patch_files, dict) and isinstance(git, dict) and git.get("dirty") is True:
        available_patches = [
            run_dir / str(file_name)
            for file_name in patch_files.values()
            if (run_dir / str(file_name)).is_file()
        ]
        if available_patches:
            st.markdown("#### Git patch files")
            for patch_path in available_patches:
                st.download_button(
                    f"Download {patch_path.name}",
                    data=patch_path.read_bytes(),
                    file_name=patch_path.name,
                    mime="text/x-diff",
                    key=f"download_{safe_key(run['run_id'])}_{safe_key(patch_path.name)}",
                )
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


def comparison_run_label(run: dict[str, Any], meta: dict[str, Any]) -> str:
    batch = (meta.get("spec", {}) or {}).get("example_batch", {})
    if isinstance(batch, dict) and batch.get("run_label"):
        return str(batch["run_label"])
    run_id = str(run["run_id"])
    short_id = run_id if len(run_id) <= 24 else f"{run_id[:19]}…{run_id[-4:]}"
    return f"{meta['name']} · {short_id}"


def _sync_compare_picker() -> None:
    selected = set_compare_run_ids(
        st.session_state.get(COMPARE_PICKER_KEY, []),
        sync_picker=False,
    )
    sync_query_params(VIEW_COMPARE, compare_run_ids=selected)


def _add_compare_run_from_picker() -> None:
    selected_label = str(st.session_state.get(COMPARE_ADD_RUN_KEY, "") or "")
    label_map = st.session_state.get(COMPARE_ADD_RUN_MAP_KEY, {})
    run_id = str(label_map.get(selected_label, selected_label) if isinstance(label_map, dict) else selected_label)
    if not run_id:
        return
    set_compare_run_ids([*normalize_compare_run_ids(st.session_state.get(COMPARE_RUN_IDS_KEY, [])), run_id])
    st.session_state[COMPARE_ADD_RUN_KEY] = ""
    sync_query_params(VIEW_COMPARE, compare_run_ids=st.session_state[COMPARE_RUN_IDS_KEY])


def _render_compare_page(
    repo_root: Path,
    runs: list[dict[str, Any]],
    auto_refresh: bool,
) -> None:
    # A fragment rerun does not execute ``main`` again, therefore refresh the
    # registry here as well as the MLflow metric histories below.
    runs = list_runs(repo_root)
    st.markdown("<div class='fx-detail-subtitle'>Runs / Compare</div>", unsafe_allow_html=True)
    top_cols = st.columns([5.1, 1.1])
    with top_cols[0]:
        st.markdown(brand_markup(suffix="Compare", level=1), unsafe_allow_html=True)
    with top_cols[1]:
        if st.button("Dashboard", key="compare_to_dashboard", use_container_width=True):
            clear_compare_selection()
            navigate_to(VIEW_DASHBOARD)
            rerun_app()

    run_map = {str(run["run_id"]): run for run in runs}
    selected_ids = [
        run_id
        for run_id in normalize_compare_run_ids(
            st.session_state.get(COMPARE_RUN_IDS_KEY, [])
        )
        if run_id in run_map
    ]
    if selected_ids != st.session_state.get(COMPARE_RUN_IDS_KEY, []):
        set_compare_run_ids(selected_ids)
    if comparison_has_active_runs(selected_ids, run_map) != auto_refresh:
        rerun_app()
        return

    metas = {
        run_id: extract_run_meta(repo_root, run)
        for run_id, run in run_map.items()
    }
    labels = {
        run_id: comparison_run_label(run, metas[run_id])
        for run_id, run in run_map.items()
    }
    selected_example_context = example_context_from_specs(
        [metas[run_id]["spec"] for run_id in selected_ids]
    )
    example_definition: ExampleDefinition | None = None
    if selected_example_context:
        try:
            example_definition = load_examples(examples_catalog_path()).get(
                str(selected_example_context.get("example_key", ""))
            )
        except (OSError, ValueError):
            example_definition = None
    available_ids = [run_id for run_id in run_map if run_id not in selected_ids]
    available_labels = {labels[run_id]: run_id for run_id in available_ids}
    st.session_state[COMPARE_ADD_RUN_MAP_KEY] = available_labels
    add_cols = st.columns([1.15, 4.85])
    with add_cols[0]:
        st.markdown("<div class='fx-selection-count'>Selected runs</div>", unsafe_allow_html=True)
    with add_cols[1]:
        st.selectbox(
            "Add run",
            options=[""] + list(available_labels),
            key=COMPARE_ADD_RUN_KEY,
            format_func=lambda label: "Add a run…" if not label else label,
            on_change=_add_compare_run_from_picker,
            label_visibility="collapsed",
        )
    if not selected_ids:
        st.info("Select at least one run to start a comparison.")
        return

    example_flash = str(st.session_state.get(EXAMPLE_FLASH_KEY, "") or "")
    if example_flash:
        st.warning(example_flash)
        st.session_state[EXAMPLE_FLASH_KEY] = ""
    if selected_example_context:
        title = (
            example_definition.title
            if example_definition is not None
            else str(selected_example_context.get("example_title", "Example comparison"))
        )
        description = (
            str(example_definition.data.get("description", ""))
            if example_definition is not None
            else "Coordinated Example runs were launched together."
        )
        st.markdown(
            (
                "<div class='fx-example-context'>"
                f"<div class='fx-example-context-title'>{escape(title)}</div>"
                f"<div class='fx-detail-subtitle'>{escape(description)}</div>"
                f"<div class='fx-detail-subtitle' style='margin-top:.35rem'>{escape(str(selected_example_context.get('run_count', len(selected_ids))))} coordinated runs</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    selected_runs = [run_map[run_id] for run_id in selected_ids]
    selected_metas = [metas[run_id] for run_id in selected_ids]
    with st.container(key="compare-selected-runs"):
        for run_id, meta in zip(selected_ids, selected_metas):
            with st.container(key=f"compare-chip-{safe_key(run_id)}"):
                chip_cols = st.columns([5.0, .8])
                with chip_cols[0]:
                    st.markdown(
                        f"**{labels[run_id]}** &nbsp; {format_status_badge(meta['status'])}",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{meta['started_at'] or meta['created_at'] or 'N/A'} · {meta['duration']}")
                with chip_cols[1]:
                    if st.button("Remove", key=f"compare_remove_{safe_key(run_id)}", use_container_width=True):
                        set_compare_run_ids([item for item in selected_ids if item != run_id])
                        sync_query_params(VIEW_COMPARE, compare_run_ids=st.session_state[COMPARE_RUN_IDS_KEY])
                        rerun_app()
    if len(selected_ids) < 2:
        st.info("Add another run above to overlay metrics and inspect configuration differences.")
        return

    all_points = []
    points_by_run: dict[str, list[Any]] = {}
    metric_messages: list[str] = []
    for run_id, run, meta in zip(selected_ids, selected_runs, selected_metas):
        mlflow_context = get_run_mlflow_context(repo_root, run, meta)
        if not mlflow_context["enabled"]:
            metric_messages.append(f"{labels[run_id]}: MLflow was not configured.")
            points_by_run[run_id] = []
            continue
        if not mlflow_context["run_id"]:
            metric_messages.append(f"{labels[run_id]}: MLflow run ID is not available yet.")
            points_by_run[run_id] = []
            continue
        result = load_metric_histories(
            mlflow_context["run_id"],
            mlflow_context["tracking_uri"],
        )
        if result.error:
            metric_messages.append(f"{labels[run_id]}: {result.error}")
        if result.metric_errors:
            metric_messages.append(
                f"{labels[run_id]}: unreadable metrics: "
                + ", ".join(sorted(result.metric_errors))
            )
        run_points = [replace(point, run_id=run_id) for point in result.points]
        points_by_run[run_id] = run_points
        all_points.extend(run_points)

    if metric_messages:
        with st.expander("Metric loading details", expanded=False):
            for message in metric_messages:
                st.caption(message)

    final_by_run = {
        run_id: {item.metric: item for item in load_final_metrics(points_by_run.get(run_id, []))}
        for run_id in selected_ids
    }
    final_metric_names = sorted(
        {
            metric
            for final_metrics in final_by_run.values()
            for metric in final_metrics
        }
    )
    st.markdown("### Final metrics comparison")
    if final_metric_names:
        final_rows: list[dict[str, str]] = []
        for metric in final_metric_names:
            row: dict[str, str] = {"Metric": metric}
            for run_id in selected_ids:
                value = final_by_run[run_id].get(metric)
                row[labels[run_id]] = format_metric_value(value.value) if value else "N/A"
            final_rows.append(row)
        render_final_metrics_table(final_rows, key="compare-final-metrics")
    else:
        st.caption("No final metrics are available yet.")

    metrics = available_metric_names(all_points)
    st.markdown("### Metric history")
    if metrics:
        metric_presence = {
            metric: sum(any(point.metric == metric for point in run_points) for run_points in points_by_run.values())
            for metric in metrics
        }
        ordered_metrics = sorted(metrics, key=lambda metric: (-metric_presence[metric], metric.lower()))
        selector_key = "compare_selected_metrics"
        context_id = ""
        if selected_example_context:
            context_id = (
                f"{selected_example_context.get('example_key', '')}:"
                f"{selected_example_context.get('group_id', '')}"
            )
        if context_id and st.session_state.get(COMPARE_METRIC_CONTEXT_KEY) != context_id:
            preferred = available_preferred_metrics(selected_example_context, ordered_metrics)
            if preferred:
                st.session_state[selector_key] = preferred
                st.session_state[COMPARE_METRIC_CONTEXT_KEY] = context_id
        elif selector_key not in st.session_state:
            st.session_state[selector_key] = ordered_metrics[: min(2, len(ordered_metrics))]
        selected_metrics = st.multiselect(
            "Metrics", options=ordered_metrics, key=selector_key,
            help="Metrics available in more selected runs are shown first.",
        )
        chart_columns = st.columns(2)
        for index, metric in enumerate(selected_metrics):
            chart_frame = metric_points_frame(
                [point for point in all_points if point.metric == metric], run_labels=labels,
            )
            with chart_columns[index % 2]:
                if chart_frame.empty or chart_frame["x"].isna().all():
                    st.info(f"{metric}: no usable step or timestamp was recorded.")
                else:
                    title_map = {
                        str(item.get("metric")): str(item.get("label"))
                        for item in (example_definition.preferred_metrics if example_definition else [])
                    }
                    render_metric_chart_card(
                        metric,
                        chart_frame,
                        compare=True,
                        display_title=title_map.get(metric),
                    )
    else:
        if selected_example_context:
            st.status("Starting example runs…", state="running", expanded=False)
            st.caption(
                f"{len(selected_ids)} runs are running in parallel. Metric charts will appear automatically as soon as MLflow begins reporting results."
            )
        else:
            st.info("No MLflow metric history is available for the selected runs.")
    if selected_example_context and all_points and any(
        not points_by_run.get(run_id) for run_id in selected_ids
    ):
        st.caption("Some Example runs are still starting; charts show the metric histories currently available.")

    st.markdown("### Run artifacts")
    artifact_run_key = "ui_compare_artifact_run"
    if st.session_state.get(artifact_run_key) not in selected_ids:
        st.session_state[artifact_run_key] = selected_ids[0]
    artifact_run_id = st.selectbox(
        "Run",
        options=selected_ids,
        key=artifact_run_key,
        format_func=lambda current_run_id: labels[current_run_id],
    )
    artifact_run = run_map[artifact_run_id]
    artifact_meta = metas[artifact_run_id]
    artifact_context = get_run_mlflow_context(repo_root, artifact_run, artifact_meta)
    if not artifact_context["enabled"]:
        st.info("MLflow was not configured for this run.")
    elif not artifact_context["run_id"]:
        st.info("MLflow run ID is not available yet.")
    else:
        render_artifact_viewer(
            artifact_context["run_id"],
            artifact_context["tracking_uri"],
            key_prefix=f"ui_compare_{safe_key(artifact_run_id)}",
        )

    st.markdown("### Configuration diff")
    show_identical = st.checkbox(
        "Show unchanged",
        key="compare_show_identical_parameters",
    )
    configs = {
        run_id: experiment_config_from_spec(metas[run_id]["spec"])
        for run_id in selected_ids
    }
    diff_rows = build_config_diff(configs)
    if not show_identical:
        diff_rows = [row for row in diff_rows if row.differs]
    if diff_rows:
        config_rows: list[dict[str, str]] = []
        for row in diff_rows:
            rendered = {"Parameter": row.parameter}
            for run_id in selected_ids:
                rendered[labels[run_id]] = row.values.get(run_id, "N/A")
            config_rows.append(rendered)
        st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)
    elif show_identical:
        st.caption("No saved configuration values are available for these runs.")
    else:
        st.caption("No differing saved configuration values.")

    provenance_rows: list[dict[str, str]] = []
    for run_id, run, meta in zip(selected_ids, selected_runs, selected_metas):
        provenance = load_provenance(Path(run["run_dir"]))
        git = provenance.get("git", {}) if isinstance(provenance, dict) else {}
        if not isinstance(git, dict):
            git = {}
        dirty = git.get("dirty")
        provenance_rows.append(
            {
                "Run": labels[run_id],
                "Status": meta["status"],
                "Commit": str(git.get("short_commit") or "N/A"),
                "Branch": str(git.get("branch") or ("Detached HEAD" if git.get("detached_head") else "N/A")),
                "Dirty": "yes" if dirty is True else "no" if dirty is False else "N/A",
            }
        )
    st.markdown("### Git summary")
    st.dataframe(pd.DataFrame(provenance_rows), use_container_width=True, hide_index=True)


def render_compare_page(repo_root: Path, runs: list[dict[str, Any]]) -> None:
    run_map = {str(run["run_id"]): run for run in runs}
    selected_ids = normalize_compare_run_ids(
        st.session_state.get(COMPARE_RUN_IDS_KEY, [])
    )
    auto_refresh = comparison_has_active_runs(selected_ids, run_map)
    run_every = "0.6s" if auto_refresh else None
    st.fragment(run_every=run_every)(_render_compare_page)(
        repo_root,
        runs,
        auto_refresh,
    )


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
    mlflow_context = get_run_mlflow_context(repo_root, run, meta)
    flash_message = str(st.session_state.get(FLASH_MESSAGE_KEY, "") or "")
    if flash_message:
        st.success(flash_message)
        st.session_state[FLASH_MESSAGE_KEY] = ""
    st.markdown(
        f"<div class='fx-detail-subtitle'>Runs / {meta['name']}</div>",
        unsafe_allow_html=True,
    )
    header_cols = st.columns([4.8, 2.8])
    with header_cols[0]:
        render_run_header(meta, run)
    with header_cols[1]:
        action_grid = st.columns(3)
        with action_grid[0]:
            if st.button("Create Run", key="run_create", use_container_width=True):
                set_create_step(CREATE_STEPS[0])
                navigate_to(VIEW_CREATE)
                rerun_app()
        with action_grid[1]:
            if st.button("Compare", key="run_compare", use_container_width=True):
                clear_compare_selection()
                set_compare_run_ids([run["run_id"]])
                navigate_to(VIEW_COMPARE, compare_run_ids=[run["run_id"]])
                rerun_app()
        with action_grid[0]:
            if st.button("Clone", key="run_clone", use_container_width=True):
                spec = meta["spec"]
                snapshot = spec.get("ui_state_snapshot")
                if snapshot:
                    restore_ui_state_snapshot(snapshot, defaults)
                set_create_step("run")
                navigate_to(VIEW_CREATE)
                rerun_app()
        with action_grid[1]:
            if st.button("Re-run", key="run_rerun", use_container_width=True):
                try:
                    new_status = rerun_saved_run(repo_root, Path(run["run_dir"]))
                except (OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Could not re-run this experiment: {exc}")
                else:
                    st.session_state[FLASH_MESSAGE_KEY] = (
                        f"Re-run started: {new_status.get('run_name', new_status['run_id'])}"
                    )
                    navigate_to(VIEW_RUN, run_id=new_status["run_id"])
                    rerun_app()
        with action_grid[2]:
            can_stop = run.get("status") == "running"
            if st.button("Stop", key="run_stop", disabled=not can_stop, use_container_width=True):
                stop_run(Path(run["run_dir"]))
                rerun_app()
        with action_grid[2]:
            mlflow_clicked = st.button(
                "MLflow",
                key="run_mlflow_open",
                disabled=not mlflow_context["enabled"],
                use_container_width=True,
            )
            if mlflow_clicked:
                try:
                    target_url = ""
                    if mlflow_context["target"] == "local":
                        base_ui_url = ensure_local_mlflow_ui(
                            repo_root,
                            mlflow_context["tracking_uri"],
                            preferred_ui_url=mlflow_context["ui_url_hint"] or DEFAULT_LOCAL_MLFLOW_UI_URL,
                        )
                        target_url = build_mlflow_run_url(
                            base_ui_url,
                            mlflow_context["experiment_id"],
                            mlflow_context["run_id"],
                        )
                        persist_mlflow_metadata(
                            Path(run["run_dir"]),
                            mlflow_context["status"],
                            mlflow_url=target_url or base_ui_url,
                            mlflow_run_id=mlflow_context["run_id"] or None,
                            mlflow_experiment_id=mlflow_context["experiment_id"] or None,
                        )
                    else:
                        target_url = (
                            mlflow_context["url"]
                            or build_mlflow_run_url(
                                normalize_mlflow_ui_url(mlflow_context["tracking_uri"]),
                                mlflow_context["experiment_id"],
                                mlflow_context["run_id"],
                            )
                        )

                    if not target_url:
                        raise RuntimeError("MLflow URL is not available for this run yet.")
                    queue_browser_open(target_url)
                except RuntimeError as exc:
                    st.error(str(exc))

    tabs = st.tabs(
        [
            "Analytics",
            "Artifacts",
            "Parameters",
            "Git",
            "Logs",
            "Journal",
            "Files",
            "Overview",
        ]
    )
    with tabs[0]:
        render_analytics_view(repo_root, run, meta)
    with tabs[1]:
        render_run_artifacts_view(repo_root, run, meta)
    with tabs[2]:
        render_parameters_view(repo_root, run, meta)
    with tabs[3]:
        render_provenance_view(run)
    with tabs[4]:
        render_logs_view(run)
    with tabs[5]:
        render_journal_view(run)
    with tabs[6]:
        render_files_view(repo_root, run)
    with tabs[7]:
        render_overview_view(repo_root, run, meta)


def main() -> None:
    st.set_page_config(
        page_title="FedXplore Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    install_keyboard_guard()
    install_history_navigation_sync()
    apply_page_styles()
    install_button_palette_hook()

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
    install_sidebar_controller(
        auto_expand=not bool(st.session_state.get(SIDEBAR_BOOTSTRAP_KEY, False))
    )
    st.session_state[SIDEBAR_BOOTSTRAP_KEY] = True

    try:
        templates = load_templates(repo_root / "ui/templates")
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(defaults)
    runs = list_runs(repo_root)
    view = st.session_state.get(VIEW_KEY, VIEW_DASHBOARD)
    if view == VIEW_CREATE:
        render_create_page(repo_root, defaults, options, templates)
    elif view == VIEW_EXAMPLES:
        render_examples_page(repo_root)
    elif view == VIEW_RUN:
        render_run_detail_page(repo_root, runs, defaults)
    elif view == VIEW_COMPARE:
        render_compare_page(repo_root, runs)
    else:
        render_dashboard_page(repo_root, runs)
    render_pending_scroll_top()
    render_pending_browser_open()


if __name__ == "__main__":
    main()
