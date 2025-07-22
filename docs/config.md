## Key Parameters

All training parameters are in `configs`. 

Main configuration file `config.yaml` contains:
- `defaults`:
    - `model`: Model architecture (e.g. `resnet18`). Add custom model in `configs/model`.
    - `dataset@train_dataset`: Client training data (`cifar10`, or custom datasets in `configs/observed_data_params`).
    - `dataset@test_dataset`: Server evaluation dataset (doesn't affect training).
    - `dataset@trust_dataset`: Additional sample stored on the server. Used by FLTrust for example.
    - `distribution`: The parameter responsible for distributing data between clients.
    - `model_trainer`: Entity responsible for technical training, eval, test model in different task domains
    - `federated_method`: Algorithm selection (`fedavg`, `fedprox`). See [method.md](method.md) for details.
    - `client_selector`: The nature of customer subset selection (e.g. `FedCBS`)
    - `manager`: Type of manager to handle multiple clients. E.g. `sequential` sequentially collects client updates (or custom manager type).
    - `losses@loss`: Client loss function, e.g. `ce` (cross-entropy) or custom loss in `configs/losses`
    - `optimizer`: Type of client optimizer, e.g. `adam`, `sgd` or any existing optimization scheme.
    - `preaggregator`: Methods for modifying client gradients before aggregation. Used in Byzantine setups.

- `random_state`: Sets reproducibility
- `training_params`
    - `batch_size`: client batch size
    - `num_workers`: amount of parallelism in `Dataloader`
    - `device`: type of computing device (`cpu` or `cuda`)
    - `device_ids` list of gpu indices (for the case of many GPUs on a machine and used many gpus for one experiment)
- `federated_params`:
    - `amount_of_clients`: Total number of clients
    - `client_subset_size`: How many clients do we select in each round
    - `communication_rounds`: Total number of rounds
    - `local_epochs`: Local client epochs
    - `client_train_val_prop`: Train/validation split on client-side to evaluate server model
    - `print_client_metrics`: Enable client metric logging
    - `server_saving_metrics`: Model selection metrics. Can be one of `"loss", "Specificity", "Sensitivity", "G-mean", "f1-score", "fbeta2-score", "ROC-AUC", "AP", "Precision (PPV)", "NPV"`
    - `server_saving_agg`: Aggregation method (`uniform` average vs `weighted` by dataset size)
    - `clients_attack_types`, `prop_attack_clients`, `attack_scheme`, `prop_attack_rounds`: see [attack.md](attacks.md) for details. 