## Logging

At the moment, two types of logging are supported:

- Base
- MLFlow

Logging is performed at the end of each round when the `log_round` function is called. If a method needs to log some specific information, you just need to override the `log_round` method (do not forget to call `super().log_round()`).

Any logger must inherit from Base and provide the following functions:

- `log_run_info` — logs basic information about the experiment, for example git information
- `log_scalar` — logs a scalar value, for example loss
- `log_pandas` — logs a pandas object with metrics
- `save_artefact` — saves a file with some useful information

### What is saved in the base setup

At the moment, the following information is collected in FedAvg:

- client data distribution
- Hydra config
- git information (commit, unstaged_changes, branch, ...)
- test loss
- test metrics (depending on the task)
- validation loss
- validation metrics
- round execution time
- client execution time statistics (max, mean, min, std)
- information about selected clients in each round (in CSV format)
- histogram of selected clients over the whole training process

This information is collected and the corresponding logger methods are called to save it. How and where this information is stored depends on the logger itself.

### Base

**FedAvg with Base Logger**

```bash
python src/train.py training_params.batch_size=32 federated_params.print_client_metrics=False training_params.device_ids=[3] federated_params.communication_rounds=3 logger=base > outputs/fedavg_base_logger.txt &
```

This type of logging does nothing. Prints remain, but no information is saved anywhere. All logger functions described above are empty and do nothing.

This is done because this logger is intended to be the base one, and in the most basic setup logging is just a txt file. All the information described above is already written to the output file via `print`, so no additional saving is required.

This logger also creates an experiment report in the Hydra folder, which can be conveniently pasted, for example, into Confluence.

It also redirects the output so that it is saved into the Hydra outputs file, and the redirection itself is only a symbolic link to that file.

### MLFlow

**FedAvg with MLFlow Logger**

```bash
python src/train.py training_params.batch_size=32 federated_params.print_client_metrics=False training_params.device_ids=[3] federated_params.communication_rounds=3 logger=mlflow > outputs/fedavg_base_logger.txt &
```

This type of logging inherits from the Base logger. Therefore, it keeps prints, creates an experiment report, and redirects the output.

In addition, it logs information to MLFlow. By default, our private server is used ([http://10.100.202.109:5000/#/experiments](http://10.100.202.109:5000/#/experiments)).

MLFlow stores experiments in the following structure: experiment_name/run_name. Both parts of the name can be configured via Hydra.

Inside MLFlow, all the data described above is saved, and plots and other visualizations are built automatically.
