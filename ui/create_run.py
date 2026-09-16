"""Pure presentation helpers for the Create Run research-workbench flow."""

from __future__ import annotations

import re
from typing import Any, Mapping


SPECIAL_LABELS = {
    "lr": "Learning rate",
    "eps": "Epsilon",
    "betas": "Betas",
    "_target_": "Hydra target",
    "random_state": "Random seed",
    "amount_of_clients": "Total clients",
    "client_subset_size": "Clients per round",
    "communication_rounds": "Communication rounds",
    "local_epochs": "Local epochs",
    "batch_size": "Batch size",
    "num_workers": "Worker processes",
    "print_client_metrics": "Print client metrics",
    "server_saving_metrics": "Save server metrics",
    "server_saving_agg": "Save aggregated metrics",
    "label_smoothing": "Label smoothing",
    "ignore_index": "Ignore index",
    "weight_decay": "Weight decay",
    "pos_weight": "Positive-class weight",
    "tracking_uri": "Tracking URI",
    "experiment_name": "Experiment name",
    "run_name": "Run name",
    "device_ids": "GPU devices",
    "client_train_val_prop": "Client train/validation split",
    "prop_attack_clients": "Malicious client fraction",
    "prop_attack_rounds": "Attacked-round fraction",
    "attack_scheme": "Attack schedule",
}


def readable_label(path: str) -> str:
    """Convert a stored config path to a concise UI label without changing it."""

    leaf = str(path).split(".")[-1]
    if leaf in SPECIAL_LABELS:
        return SPECIAL_LABELS[leaf]
    words = re.sub(r"[_-]+", " ", leaf).strip()
    return words.capitalize() if words else str(path)


def readable_option(value: Any) -> str:
    """Make config option identifiers presentable while retaining their meaning."""

    rendered = str(value or "").strip()
    if not rendered:
        return "Not configured"
    return re.sub(r"[_-]+", " ", rendered).title()


def build_experiment_summary(state: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return only high-signal experiment settings from the live form state."""

    def value(key: str, fallback: str = "") -> Any:
        return state.get(key, fallback)

    rows: list[tuple[str, str]] = []
    method = value("ui_federated_method")
    if method:
        rows.append(("Method", readable_option(method)))
    selector = value("ui_client_selector")
    if selector:
        rows.append(("Client selection", readable_option(selector)))

    dataset = value("ui_train_dataset")
    if dataset:
        roles = [readable_option(dataset)]
        test_dataset = value("ui_test_dataset")
        if test_dataset and test_dataset != dataset:
            roles.append(f"Test: {readable_option(test_dataset)}")
        trust_dataset = value("ui_trust_dataset")
        if trust_dataset:
            roles.append(f"Trust: {readable_option(trust_dataset)}")
        rows.append(("Dataset", " · ".join(roles)))

    clients = value("ui_base__federated_params_amount_of_clients")
    subset = value("ui_base__federated_params_client_subset_size")
    if clients or subset:
        detail = str(clients or "?")
        if subset:
            detail += f" clients · {subset}/round"
        else:
            detail += " clients"
        rows.append(("Federation", detail))

    distribution = value("ui_distribution")
    if distribution:
        distribution_detail = readable_option(distribution)
        alpha = value("ui_comp__distribution__dirichlet__alpha")
        if alpha not in (None, "") and "dirichlet" in str(distribution).lower():
            distribution_detail += f" α = {alpha}"
        rows.append(("Distribution", distribution_detail))

    for label, key in [("Optimizer", "ui_optimizer")]:
        option = value(key)
        if option:
            rows.append((label, readable_option(option)))

    learning_rate = value(f"ui_comp__optimizer__{value('ui_optimizer')}__lr")
    if learning_rate not in (None, "") and rows and rows[-1][0] == "Optimizer":
        rows[-1] = ("Optimizer", f"{rows[-1][1]} · lr={learning_rate}")

    attack = value("ui_attack_type", "no_attack")
    attack_detail = readable_option(attack)
    if str(attack) != "no_attack":
        fraction = value("ui_base__federated_params_prop_attack_clients")
        if fraction not in (None, "", 0, 0.0):
            attack_detail += f" · {float(fraction):.0%} malicious"
    rows.append(("Attack", attack_detail))

    preaggregator = value("ui_preaggregator")
    if preaggregator:
        rows.append(("Pre-aggregation", readable_option(preaggregator)))
    model = value("ui_model")
    if model:
        rows.append(("Training", readable_option(model)))
    logger = value("ui_logger")
    if logger:
        rows.append(("Tracking", readable_option(logger)))
    device = value("ui_device_mode")
    if device:
        device_ids = value("ui_device_ids_selected", [])
        suffix = ""
        if str(device).lower() == "cuda" and device_ids:
            suffix = " · " + ", ".join(f"GPU {item}" for item in device_ids)
        rows.append(("Runtime", f"{str(device).upper()}{suffix}"))
    seed = value("ui_base__random_state")
    if seed not in (None, ""):
        rows.append(("Seed", str(seed)))
    return rows


def validate_experiment_state(
    state: Mapping[str, Any], *, requires_trust_dataset: bool = False
) -> tuple[list[str], list[str]]:
    """Return launch-blocking errors and non-blocking research warnings."""

    def number(key: str) -> float | None:
        try:
            return float(state.get(key))
        except (TypeError, ValueError):
            return None

    errors: list[str] = []
    warnings: list[str] = []
    total = number("ui_base__federated_params_amount_of_clients")
    subset = number("ui_base__federated_params_client_subset_size")
    rounds = number("ui_base__federated_params_communication_rounds")
    epochs = number("ui_base__federated_params_local_epochs")
    if total is None or total <= 0:
        errors.append("Total clients must be greater than zero.")
    if subset is None or subset <= 0:
        errors.append("Clients per round must be greater than zero.")
    if total is not None and subset is not None and subset > total:
        errors.append("Clients per round cannot exceed total clients.")
    if rounds is None or rounds <= 0:
        errors.append("Communication rounds must be greater than zero.")
    if epochs is None or epochs <= 0:
        errors.append("Local epochs must be greater than zero.")
    if requires_trust_dataset and not state.get("ui_trust_dataset"):
        warnings.append("The selected FL method requires a server-side trust dataset.")
    if state.get("ui_attack_type", "no_attack") != "no_attack":
        malicious_fraction = number("ui_base__federated_params_prop_attack_clients")
        if malicious_fraction is None or malicious_fraction <= 0:
            warnings.append("An attack is selected but the malicious-client fraction is zero.")
    return errors, warnings


def initial_dataset_roles(
    selected_dataset: str, previous_base_dataset: str
) -> dict[str, str] | None:
    """Return one intentional role initialization, never a continuous sync."""

    if not selected_dataset or selected_dataset == previous_base_dataset:
        return None
    return {
        "train_dataset": selected_dataset,
        "test_dataset": selected_dataset,
        "trust_dataset": "",
    }
