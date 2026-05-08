import re
import os
import argparse
import subprocess
import time
import shlex

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(description="Run federated learning experiments.")
parser.add_argument(
    "--max_parallel",
    type=int,
    default=2,
    help="Maximum number of parallel runs",
)
args = parser.parse_args()

# =========================
# Paths
# =========================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Configuration
# =========================
DATASETS = ["cifar10"]

PREAGGREGATORS = [None, "fbm"]

BYZANT_METHODS = ["central_clip"]
FEDERATED_METHODS = BYZANT_METHODS
GPU_IDS = [0, 1, 3]

CLIENT_SELECTORS = ["pow"]
# ATTACK_TYPES = ["no_attack", "label_flip", "sign_flip", "random_grad", "alie"]
ATTACK_TYPES = ["random_grad", "alie"]

experiment_name = "FL_PPBC_BYZ_PAPER_02_prop_5_clients_hetero_05_pow"

distribution = "dirichlet"
dir_alpha = 0.5

attack_prop = 0.2
attack_params = (
    f"federated_params.attack_scheme=constant "
    f"federated_params.prop_attack_rounds=1.0 "
    f"federated_params.prop_attack_clients={attack_prop}"
)
num_rounds = 750
client_subset_size = 5
model = "resnet18"
trainer = "image"


# =========================
# Helpers
# =========================
def rotate_list(lst, k):
    k = k % len(lst)
    return lst[k:] + lst[:k]


def run_experiment(cmd_list, run_name, output_file):
    print(f"[START] {run_name}", flush=True)
    with open(output_file, "w", buffering=1) as log_f:
        try:
            p = subprocess.Popen(cmd_list, stdout=log_f, stderr=subprocess.STDOUT)
            return p, run_name
        except Exception as e:
            print(f"[ERROR] Failed to start {run_name}: {e}", flush=True)
            return None, run_name


# =========================
# Build all commands
# =========================
jobs = []
job_idx = 0

for method in FEDERATED_METHODS:
    for preaggregator in PREAGGREGATORS:
        for attack in ATTACK_TYPES:
            for client_selector in CLIENT_SELECTORS:
                for dataset in DATASETS:
                    # Preaggregator is allowed only with central_clip
                    if preaggregator is not None and method != "central_clip":
                        continue

                    optimizer = (
                        "sgd" if method in ["recess", "central_clip"] else "adam"
                    )
                    run_name = f"{method if preaggregator is None else method + '_' + preaggregator}_{attack}_{attack_prop}_prop_{client_selector}_round_{num_rounds}"
                    output_file = f"{OUTPUT_DIR}/{run_name}.txt"

                    # GPU rotation
                    client_gpus = rotate_list(GPU_IDS, job_idx + 1)
                    gpu_ids_str = "[" + ",".join(map(str, client_gpus)) + "]"
                    job_idx += 1

                    # Build command string
                    cmd = f"""
                        python ./src/train.py
                        federated_method={method}
                        client_selector={client_selector}
                        dataset@train_dataset={dataset}
                        dataset@test_dataset={dataset}
                        dataset@trust_dataset={dataset}
                        model={model}
                        federated_params.clients_attack_types={attack}
                        {attack_params if attack != "no_attack" else ""}
                        {"preaggregator=" + preaggregator if preaggregator else ""}
                        model_trainer={trainer}
                        optimizer={optimizer}
                        manager.batch_generator.batch_size=5
                        training_params.batch_size=64
                        distribution={distribution}
                        {f'distribution.alpha={dir_alpha}' if distribution == "dirichlet" else ""}
                        federated_params.print_client_metrics=False
                        federated_params.communication_rounds={num_rounds}
                        federated_params.amount_of_clients=25
                        federated_params.client_subset_size={client_subset_size}
                        logger=mlflow
                        logger.experiment_name={experiment_name}
                        logger.run_name={run_name}
                        training_params.device_ids={gpu_ids_str}
                        {"+trust_dataset.num_trust_samples=2500" if method == "fltrust" else ""}
                    """

                    # Normalize spaces
                    cmd = re.sub(r"\s+", " ", cmd).strip()

                    cmd_list = shlex.split(cmd)
                    jobs.append((cmd_list, run_name, output_file))

# =========================
# Execute with limited parallelism
# =========================
print(f"Total runs: {len(jobs)}, Max parallel: {args.max_parallel}\n")

max_parallel = args.max_parallel
running = []
job_idx = 0

while job_idx < len(jobs) or running:
    # Start new proccesses
    while job_idx < len(jobs) and len(running) < max_parallel:
        cmd_list, run_name, output_file = jobs[job_idx]
        proc, name = run_experiment(cmd_list, run_name, output_file)
        if proc:
            running.append((proc, name))
        job_idx += 1
        time.sleep(5)

    time.sleep(10)

    # Check ended processes
    still_running = []
    for p, name in running:
        ret = p.poll()
        if ret is None:
            still_running.append((p, name))
        else:
            print(f"[{'OK' if ret == 0 else 'FAIL'}] {name} (code={ret})", flush=True)
    running = still_running

print("All experiments finished.")
