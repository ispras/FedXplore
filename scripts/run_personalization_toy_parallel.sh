#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/skorik/federated_research/venv/bin/python}
RUN_DIR=${RUN_DIR:-outputs/personalization_toy_parallel/$(date +%Y%m%d_%H%M%S)}
SEED=${SEED:-42}
ROUNDS=${ROUNDS:-20}
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
  logger=base
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
  local name=$1
  shift
  "$PYTHON" src/train.py \
    "${COMMON[@]}" \
    "$@" \
    "hydra.run.dir=$RUN_DIR/${name}_hydra" \
    > "$RUN_DIR/$name.txt" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

start_time=$(date +%s)
printf 'Run directory: %s\nRounds: %s\nDevice: %s\n' "$RUN_DIR" "$ROUNDS" "$DEVICE"
printf 'Seed: %s\nManager batch size: %s\n' "$SEED" "$MANAGER_BATCH_SIZE"
printf 'Ditto proximity: %s\n' "$DITTO_PROXIMITY"
launch fedavg federated_method=fedavg
launch personalized_local federated_method=personalization_toy federated_method.proximity=0.0
launch ditto_fixed federated_method=ditto_fixed \
  "federated_method.proximity=$DITTO_PROXIMITY"
launch pfedme_fixed federated_method=pfedme_fixed
launch fedrep_fixed federated_method=fedrep_fixed
launch fedamp_fixed federated_method=fedamp_fixed

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
