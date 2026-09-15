#!/usr/bin/env bash

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
LOGGER=${LOGGER:-mlflow}
RUN_GROUP=${RUN_GROUP:-interdependency_toy_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-outputs/interdependency_toy_parallel/$RUN_GROUP}
TRACKING_URI=${TRACKING_URI:-$PWD/outputs/mlruns}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-FedXplore Toy Examples}
SEED=${SEED:-42}
GPU_IDS=${GPU_IDS:-0}
DEVICE=${DEVICE:-cuda}
ROUNDS=${ROUNDS:-40}
ATTACK_PROPORTION=${ATTACK_PROPORTION:-0.35}
CANDIDATE_SET_SIZE=${CANDIDATE_SET_SIZE:-15}
MANAGER_BATCH_SIZE=${MANAGER_BATCH_SIZE:-10}
TAU_CLIP=${TAU_CLIP:-0.2}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

mkdir -p "$RUN_DIR"

COMMON=(
  "random_state=$SEED"
  dataset@train_dataset=synthetic_2d
  dataset@test_dataset=synthetic_2d
  model=logistic_regression
  model_trainer=image
  "logger=$LOGGER"
  optimizer=sgd
  optimizer.lr=0.1
  distribution=dirichlet
  distribution.alpha=2.0
  training_params.batch_size=64
  training_params.num_workers=0
  "training_params.device=$DEVICE"
  "manager.batch_generator.batch_size=$MANAGER_BATCH_SIZE"
  federated_params.amount_of_clients=20
  federated_params.client_subset_size=10
  "federated_params.communication_rounds=$ROUNDS"
  federated_params.print_client_metrics=False
  federated_params.server_saving_metrics=[]
)

if [[ "$DEVICE" == cuda ]]; then
  COMMON+=("training_params.device_ids=[$GPU_IDS]")
else
  COMMON+=("training_params.device_ids=[]")
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
      "+logger.tags.toy_suite=interdependency"
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
  printf '%s\t%s\n' "$!" "$condition" >> "$RUN_DIR/pids.tsv"
}

start_time=$(date +%s)
printf 'Run directory: %s\nDevice: %s\nGPU IDs: %s\nRounds: %s\n' \
  "$RUN_DIR" "$DEVICE" "$GPU_IDS" "$ROUNDS"
printf 'Attack proportion: %s\nCandidate set size: %s\nManager batch size: %s\n' \
  "$ATTACK_PROPORTION" "$CANDIDATE_SET_SIZE" "$MANAGER_BATCH_SIZE"
printf 'CentralClip tau: %s\n' "$TAU_CLIP"
printf 'Logger: %s\nRun group: %s\nSeed: %s\n' "$LOGGER" "$RUN_GROUP" "$SEED"
printf 'Thread limits: OMP=%s MKL=%s OpenBLAS=%s NumExpr=%s\n' \
  "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" \
  "$NUMEXPR_NUM_THREADS"

launch clean_cc_uniform central_clip \
  federated_method=central_clip "federated_method.tau_clip=$TAU_CLIP" \
  client_selector=uniform \
  federated_params.clients_attack_types=no_attack \
  federated_params.prop_attack_clients=0.0 \
  federated_params.attack_scheme=no_attack \
  federated_params.prop_attack_rounds=0.0

launch labelflip_cc_uniform central_clip \
  federated_method=central_clip "federated_method.tau_clip=$TAU_CLIP" \
  client_selector=uniform \
  federated_params.clients_attack_types=binary_label_flip \
  "federated_params.prop_attack_clients=$ATTACK_PROPORTION" \
  federated_params.attack_scheme=constant \
  federated_params.prop_attack_rounds=1.0

launch labelflip_cc_pow central_clip \
  federated_method=central_clip "federated_method.tau_clip=$TAU_CLIP" \
  client_selector=pow "client_selector.candidate_set_size=$CANDIDATE_SET_SIZE" \
  federated_params.clients_attack_types=binary_label_flip \
  "federated_params.prop_attack_clients=$ATTACK_PROPORTION" \
  federated_params.attack_scheme=constant \
  federated_params.prop_attack_rounds=1.0

launch labelflip_cc_fedcbs central_clip \
  federated_method=central_clip "federated_method.tau_clip=$TAU_CLIP" \
  client_selector=fedcbs \
  federated_params.clients_attack_types=binary_label_flip \
  "federated_params.prop_attack_clients=$ATTACK_PROPORTION" \
  federated_params.attack_scheme=constant \
  federated_params.prop_attack_rounds=1.0

launch labelflip_fedavg_uniform fedavg \
  federated_method=fedavg \
  client_selector=uniform \
  federated_params.clients_attack_types=binary_label_flip \
  "federated_params.prop_attack_clients=$ATTACK_PROPORTION" \
  federated_params.attack_scheme=constant \
  federated_params.prop_attack_rounds=1.0

printf 'Started %s runs.\n' "${#pids[@]}"

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf '[OK] %s\n' "${names[$index]}"
  else
    printf '[FAIL] %s\n' "${names[$index]}"
    failed=1
  fi
done

elapsed=$(( $(date +%s) - start_time ))
printf 'All runs finished in %s seconds.\n' "$elapsed"
exit "$failed"
