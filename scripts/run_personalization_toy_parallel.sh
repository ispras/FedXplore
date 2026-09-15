#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
LOGGER=${LOGGER:-mlflow}
RUN_GROUP=${RUN_GROUP:-personalization_toy_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-outputs/personalization_toy_parallel/$RUN_GROUP}
TRACKING_URI=${TRACKING_URI:-$PWD/outputs/mlruns}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-FedXplore Toy Examples}
SEED=${SEED:-42}
ROUNDS=${ROUNDS:-30}
DEVICE=${DEVICE:-cpu}
MANAGER_BATCH_SIZE=${MANAGER_BATCH_SIZE:-5}
DITTO_PROXIMITY=${DITTO_PROXIMITY:-1.0}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

mkdir -p "$RUN_DIR"

COMMON=(
  "random_state=$SEED"
  dataset@train_dataset=personalization_2d
  dataset@test_dataset=personalization_2d
  model=factorized_linear
  model_trainer=image
  "logger=$LOGGER"
  optimizer=sgd
  optimizer.lr=0.08
  distribution=uniform
  training_params.batch_size=64
  training_params.num_workers=0
  "training_params.device=$DEVICE"
  "manager.batch_generator.batch_size=$MANAGER_BATCH_SIZE"
  federated_params.amount_of_clients=10
  federated_params.client_subset_size=10
  "federated_params.communication_rounds=$ROUNDS"
  federated_params.local_epochs=1
  federated_params.client_train_val_prop=0.25
  federated_params.print_client_metrics=False
  federated_params.server_saving_metrics=[]
)

if [[ "$DEVICE" == cuda ]]; then
  COMMON+=(training_params.device_ids=[0])
else
  COMMON+=(training_params.device_ids=[])
fi

pids=()
names=()

stop_children() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}

trap stop_children INT TERM

launch() {
  local condition=$1
  local method=$2
  shift 2
  local logger_overrides=()
  if [[ "$LOGGER" == mlflow ]]; then
    logger_overrides=(
      "logger.tracking_uri=$TRACKING_URI"
      "logger.experiment_name=$EXPERIMENT_NAME"
      "logger.run_name=$RUN_GROUP/$condition"
      "+logger.tags.toy_suite=personalization"
      "+logger.tags.run_group=$RUN_GROUP"
      "+logger.tags.condition=$condition"
      "+logger.tags.method=$method"
      "+logger.tags.seed=$SEED"
    )
  fi
  "$PYTHON" src/train.py \
    "${COMMON[@]}" \
    "${logger_overrides[@]}" \
    "$@" \
    "hydra.run.dir=$RUN_DIR/${condition}_hydra" \
    > "$RUN_DIR/$condition.txt" 2>&1 &
  pids+=("$!")
  names+=("$condition")
}

start_time=$(date +%s)
printf 'Run directory: %s\nRounds: %s\nDevice: %s\n' "$RUN_DIR" "$ROUNDS" "$DEVICE"
printf 'Seed: %s\nManager batch size: %s\n' "$SEED" "$MANAGER_BATCH_SIZE"
printf 'Ditto proximity: %s\n' "$DITTO_PROXIMITY"
printf 'Logger: %s\nRun group: %s\n' "$LOGGER" "$RUN_GROUP"
launch fedavg fedavg federated_method=fedavg
launch personalized_local ditto federated_method=ditto federated_method.proximity=0.0
launch ditto ditto federated_method=ditto \
  "federated_method.proximity=$DITTO_PROXIMITY"
launch pfedme pfedme federated_method=pfedme
launch fedrep fedrep federated_method=fedrep
launch fedamp fedamp federated_method=fedamp

failed=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}"
  if grep -q "Shutdown clients, federated learning end" \
    "$RUN_DIR/${names[$index]}.txt"; then
    printf '[OK] %s\n' "${names[$index]}"
  else
    printf '[FAIL] %s\n' "${names[$index]}"
    failed=1
  fi
done

printf 'All runs finished in %s seconds.\n' "$(( $(date +%s) - start_time ))"
exit "$failed"
