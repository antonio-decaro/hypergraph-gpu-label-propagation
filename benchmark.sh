#!/usr/bin/env bash
set -euo pipefail

RUNS=5
LABEL_SEED=42
LABEL_CLASSES_LIST=(10 100 1000)
JSON_DIR="data"
BUILD_DIR="build"
LOG_DIR="log"

mkdir -p "$LOG_DIR"

readarray -t JSON_FILES < <(find "$JSON_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if [ ${#JSON_FILES[@]} -eq 0 ]; then
  echo "No JSON files found in $JSON_DIR" >&2
  exit 1
fi

readarray -t EXECUTABLES < <(find "$BUILD_DIR" -maxdepth 1 -type f -executable -name 'label_propagation_*' | sort)
if [ ${#EXECUTABLES[@]} -eq 0 ]; then
  echo "No label_propagation executables found in $BUILD_DIR" >&2
  exit 1
fi

timestamp() {
  date '+%Y%m%d-%H%M%S'
}

run_experiment() {
  local exe_path="$1"
  local json_path="$2"
  local dataset_name="$3"
  local run_idx="$4"
  local seed="$5"
  local label_classes="$6"
  local log_file="$7"

  local exe_name
  exe_name=$(basename "$exe_path")

  echo "[${count}/${total}] Running $exe_name on $dataset_name (run $run_idx/$RUNS, labels $label_classes)" | tee -a "$log_file"

  "$exe_path" \
    --load "$json_path" \
    --label-seed "$seed" \
    --label-classes "$label_classes" \
    --iterations 100 \
    --tolerance 1e-6 >> "$log_file" 2>&1
}

echo "Running benchmarks for ${#EXECUTABLES[@]} implementations on ${#JSON_FILES[@]} datasets across ${#LABEL_CLASSES_LIST[@]} label-class settings"

total=$(( ${#EXECUTABLES[@]} * ${#JSON_FILES[@]} * ${#LABEL_CLASSES_LIST[@]} * RUNS ))
count=0

for exe_path in "${EXECUTABLES[@]}"; do
  exe_name=$(basename "$exe_path")
  for json_path in "${JSON_FILES[@]}"; do
    dataset_name=$(basename "$json_path" .json)

    for label_classes in "${LABEL_CLASSES_LIST[@]}"; do
      log_file="$LOG_DIR/${exe_name}_${dataset_name}_seed${LABEL_SEED}_labels${label_classes}.log"
      : > "$log_file"

      for run_idx in $(seq 1 "$RUNS"); do
        count=$((count + 1))
        run_experiment "$exe_path" "$json_path" "$dataset_name" "$run_idx" "$LABEL_SEED" "$label_classes" "$log_file"
      done
    done
  done

done

echo "Benchmarking completed. Logs saved under $LOG_DIR"
