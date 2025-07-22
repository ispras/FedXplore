# ⚙️ Experiments setup

⚠️ **Important**: Follow the [Prerequisites](../README.md#-prerequisites) steps to set up enviroment. And going into `scripts` dir to run experiments.

⚠️ **Script argument**: all `scripts` have a `--device_id`, which defines the GPU idx.

## Byzantines Attacks with CIFAR-10 

```bash
python byz_cifar10_script.py > byz_cifar10_log_script.txt &
```

## Client Selection with CIFAR-10

```bash
python cs_cifar10_script.py > cs_cifar10_log_script.txt &
```

## Personalization with CIFAR-10

```bash
python pers_cifar10_script.py > pers_cifar10_log_script.txt &
```

## All available experiments with framework

```bash
python all_script.py > all_log_script.txt &
```