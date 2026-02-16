# FedXplore - Framework for Federated Learning Attacks, Defences, Client Selection and Personalization

## Table of contents
0. [Quickstart](#-quickstart-guide) -- Follow the instructions and get the result!
1. [Attacks and Defences](docs/attacks_and_defences.md) -- Deep dive into Byzantine-Robust Federated Learning
2. [Personalization](docs/personalization.md) -- Deep dive into Personalized Federated Learning
3. [Client Selection](docs/client_selection.md) -- Deep dive into Client Selection Strategies
4. [Byzantine Robustness and Client Selection](docs/interaction.md) -- Feel the flexibility of framework in modular interaction
5. [C4 notation](docs/C4.md) -- Context Container Component Code scheme.
6. [Federated Method Explaining](docs/method.md) -- Get the basis and write your own method
7. [Attacks](docs/attacks.md) -- Get the basis and write custom attack

## 🚀 Quickstart Guide
### 📋 Prerequisites
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### ⚙️ Experiment Setups

See allowed optionalization in [config.md](docs/config.md)

#### 🔄 [Federated Averaging](https://arxiv.org/pdf/1602.05629) on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  training_params.device_ids=[0] \
  > fedavg_cifar.txt
```

`device_ids` controls the GPU number (if there are several GPUs on the machine). You can specify multiple ids, then the training will be evenly distributed across the specified devices.

Additionally, `manager.batch_size` client processes will be created. To forcefully terminate the training, kill any of the processes.

#### 🌪️ Dirichlet Partition with $\alpha=0.1$ (strong heterogeneity) and [FedCor](https://arxiv.org/abs/2103.13822) client strategy

```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  train_dataset.alpha=0.1 \
  federated_params.amount_of_clients=100 \
  client_selector=fedcor \
  > fedavg_fedcor_cifar10_dirichlet_alpha0.1.txt
```

#### 🦠 [FLTrust](https://arxiv.org/abs/2012.13995) with Label Flipping Attack on [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset

```bash
python src/train.py \
  federated_method=fltrust \
  dataset@train_dataset=ptbxl \
  dataset@test_dataset=ptbxl \
  dataset@trust_dataset=ptbxl \
  model_trainer=ptbxl \
  distribution=uniform \
  model=resnet1d18 \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  federated_params.clients_attack_types=label_flip \
  federated_params.prop_attack_clients=0.5 \
  federated_params.attack_scheme=constant \
  federated_params.prop_attack_rounds=1.0 \
  > fltrust_ptbxl_label_flip_half_byzantines.txt
```

#### 🧑‍🤝‍🧑 [FedAMP](https://arxiv.org/abs/2007.03797) with 10 clusters on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset

```bash
python src/train.py \
  federated_method=fedamp \
  federated_method.strategy=sharded \
  federated_method.cluster_params=[10,0.5] \
  federated_params.amount_of_clients=100 \
  federated_params.client_subset_size=100 \
  training_params.batch_size=32 \
  > fedamp_10_clusters_cifar10.txt
```