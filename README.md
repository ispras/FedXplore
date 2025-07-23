
# FedXplore - extensible framework for federated learning

## Table of contents
0. [Running Experiments](docs/experiments.md) -- Run the command to get reproducibility
1. [Quickstart](#-quickstart-guide) -- Follow the instructions and get the result!
2. [C4 notation](docs/C4.md) -- Context Container Component Code scheme.
3. [Federated Method Explaining](docs/method.md) -- Get the basis and write your own method
4. [Config Explaining](docs/config.md) -- See allowed optionalization
5. [Attacks](docs/attacks.md) -- Get the basis and write custom attack

## 🚀 Quickstart Guide
### 📋 Prerequisites
1. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### ⚙️ Experiment Setups

#### 🔄 Standard [Federated Averaging](https://arxiv.org/pdf/1602.05629) on CIFAR-10
```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  training_params.device_ids=[0] \
  > test_run_fedavg_cifar.txt
```

`device_ids` controls the GPU number (if there are several GPUs on the machine). You can specify multiple ids, then the training will be evenly distributed across the specified devices.

Additionally, `manager.batch_size` client processes will be created. To forcefully terminate the training, kill any of the processes.

#### 🌪️ Heterogeneous CIFAR10 Experiment

**Dirichlet Partition with $\alpha=0.1$ (strong heterogeneity)**

```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  train_dataset.alpha=0.1 \
  federated_params.amount_of_clients=100 \
  > test_run_fedavg_cifar_dirichlet_strong_heterogeneity_100_clients.txt
```

**Uniform Distribution ($\alpha=1000$) with various amount_of_clients**
```bash
dataset.alpha=1000 \
federated_params.amount_of_clients=42 \
```

#### 🦠 Byzantine Attacks 

**FedAvg with Label Flipping Attack**

```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  federated_params.clients_attack_types=label_flip \
  federated_params.prop_attack_clients=0.5 \
  federated_params.attack_scheme=constant \
  federated_params.prop_attack_rounds=1.0 \
  > test_run_fedavg_cifar_label_flip_half_byzantines.txt
```
