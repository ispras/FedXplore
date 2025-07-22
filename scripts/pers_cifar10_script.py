import re
import os
import argparse
import subprocess

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

parser = argparse.ArgumentParser(description="Run federated learning experiments.")
parser.add_argument(
    "--device_id",
    type=int,
    default=0,
    help="GPU device IDx to use (default: 0)",
)
args = parser.parse_args()

# Configuration parameters
DATASETS = [
    "cifar10",
]

PREAGGREGATORS = [
    None,
]
PERSONALIZED_METHODS = [
    "personalized",
    "fedamp",
    "fedrep",
    "ditto",
    "pfedme",
]
FEDERATED_METHODS = PERSONALIZED_METHODS
CLIENT_SELECTORS = [
    "uniform",
]

ATTACK_TYPES = [
    "no_attack",
]

BASE_PARAMS = [
    "optimizer=sgd",
    "distribution=uniform",
    "manager.batch_size=5",
    "training_params.batch_size=32",
    "federated_params.print_client_metrics=False",
    "federated_params.communication_rounds=2",
    "federated_params.amount_of_clients=10",
    "federated_params.client_subset_size=5",
    f"training_params.device_ids=[{args.device_id}]",
]


attack_params = "federated_params.attack_scheme=constant federated_params.prop_attack_rounds=1.0 federated_params.prop_attack_clients=0.4"
# Run experiments
for preaggregator in PREAGGREGATORS:
    for dataset in DATASETS:
        for attack in ATTACK_TYPES:
            for client_selector in CLIENT_SELECTORS:
                for method in FEDERATED_METHODS:
                    model = "resnet18"
                    trainer = "cifar10"


                    cmd_str = f"""nohup python ../src/train.py  
                    federated_method={method}  
                    client_selector={client_selector}  
                    dataset@train_dataset={dataset}  
                    dataset@test_dataset={dataset} 
                    dataset@trust_dataset={dataset}  
                    model={model}  
                    federated_params.clients_attack_types={attack} 
                    {("preaggregator=" + str(preaggregator)) if  preaggregator is not None else ""} 
                    model_trainer={trainer} 
                    optimizer=sgd 
                    manager.batch_size=5 
                    training_params.batch_size=32 
                    distribution=uniform 
                    federated_params.print_client_metrics=False 
                    federated_params.communication_rounds=1 
                    federated_params.amount_of_clients=10 
                    federated_params.client_subset_size={5 if method not in PERSONALIZED_METHODS else 10}
                    training_params.device_ids=[{args.device_id}] > 
                    outputs/{method}_{client_selector}_{attack}_{dataset}.txt"""

                    cmd_str = cmd_str.replace("\n", " ").replace("\t", " ")
                    cmd_str = re.sub(r'\s+', ' ', cmd_str).strip()
                    # Print and execute
                    print(
                        f"Running setup: {method}_{client_selector}_{attack}_{dataset}",
                        flush=True,
                    )
                    print(f"Command is: {cmd_str}\n", flush=True)
                    subprocess.run(cmd_str, shell=True, check=False)
