#!/usr/bin/env bash
#PBS -N HLP 
#PBS -A 
#PBS -q debug
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:15:00
#PBS -l filesystems=home
#PBS -j oe
#PBS -o out.txt
#PBS -e err.txt 

if [ -n "${PBS_O_WORKDIR:-}" ]; then
  echo "Changing to working directory: $PBS_O_WORKDIR"
  cd "$PBS_O_WORKDIR"
fi

set -euo pipefail

RUNS=5
LABEL_SEED=42
LABEL_CLASSES_LIST=(10)
JSON_DIR="data"
BUILD_DIR="build"
LOG_DIR="log"
METRICS_DIR=""
RUN_EXPERIMENT=true
COLLECT_METRICS=""

# Target vendor configuration for workgroup sizing
DEFAULT_TARGET_VENDOR="nvidia"
# Map implementation names (from resolve_exe_name) to vendors when they differ from the default.
# Example: EXEC_VENDOR_OVERRIDES[openmp]="intel"
declare -A EXEC_VENDOR_OVERRIDES=()

# Workgroup sizes per vendor
declare -A WORKGROUP_SIZES=(
  [nvidia]=256
  [amd]=512
  [intel]=64
)

# Profiler configuration (edit here to change binaries or default flags)
declare -A PROFILER_BINARIES=(
  [nvidia]="ncu"
  [amd]="rocprof"
  [intel]="unitrace"
)

declare -A PROFILER_ARGS=(
  [nvidia]=""
  [amd]=""
  [intel]=""
)

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --collect-metrics {nvidia|amd|intel}  Collect GPU metrics (default: disabled)
  --metrics-dir PATH                    Directory for profiler outputs (default: LOG_DIR/metrics)
  --skip-run                            Skip executing the benchmark binaries; only collect metrics
  -h, --help                            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect-metrics)
      COLLECT_METRICS="${2:-}"
      shift 2
      ;;
    --metrics-dir)
      METRICS_DIR="${2:-}"
      shift 2
      ;;
    --skip-run)
      RUN_EXPERIMENT=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac

done

if [[ -z "$METRICS_DIR" ]]; then
  METRICS_DIR="${LOG_DIR}/metrics"
fi

mkdir -p "$LOG_DIR" "$METRICS_DIR"

timestamp() {
  date '+%Y%m%d-%H%M%S'
}

resolve_exe_name() {
  local exe_basename
  exe_basename=$(basename "$1")
  case "$exe_basename" in
    label_propagation_sycl*) echo "sycl" ;;
    label_propagation_openmp*) echo "openmp" ;;
    label_propagation_kokkos*) echo "kokkos" ;;
    *)
      echo "Unsupported executable name: $exe_basename" >&2
      exit 1
      ;;
  esac
}

resolve_exe_vendor() {
  local exe_path="$1"
  local exe_name
  exe_name=$(resolve_exe_name "$exe_path")
  local vendor="${EXEC_VENDOR_OVERRIDES[$exe_name]-}"

  if [[ -z "$vendor" ]]; then
    vendor="$DEFAULT_TARGET_VENDOR"
  fi

  if [[ -z "$vendor" ]]; then
    echo "Unable to determine target vendor for $exe_name. Set DEFAULT_TARGET_VENDOR or EXEC_VENDOR_OVERRIDES." >&2
    exit 1
  fi

  echo "${vendor,,}"
}

resolve_workgroup_size() {
  local vendor="${1,,}"
  local size="${WORKGROUP_SIZES[$vendor]-}"
  if [[ -z "$size" ]]; then
    echo "Unsupported vendor for workgroup size: $vendor" >&2
    exit 1
  fi
  echo "$size"
}

build_run_command() {
  local -n _out=$1
  local exe_path="$2"
  local json_path="$3"
  local seed="$4"
  local label_classes="$5"
  local vendor="${6:-}"

  if [[ -z "$vendor" ]]; then
    vendor=$(resolve_exe_vendor "$exe_path")
  else
    vendor="${vendor,,}"
  fi

  local workgroup_size
  workgroup_size=$(resolve_workgroup_size "$vendor")

  _out=(
    "$exe_path"
    --load "$json_path"
    --label-seed "$seed"
    --label-classes "$label_classes"
    --iterations 100
    --workgroup-size "$workgroup_size"
    --tolerance 1e-6
  )
}

resolve_profiler_binary() {
  local vendor="$1"
  local bin="${PROFILER_BINARIES[$vendor]-}"
  if [[ -z "$bin" ]]; then
    return 1
  fi
  echo "$bin"
}

append_profiler_args() {
  local vendor="$1"
  local -n cmd_ref=$2
  local args="${PROFILER_ARGS[$vendor]-}"
  if [[ -n "$args" ]]; then
    read -r -a extra <<<"$args"
    cmd_ref+=("${extra[@]}")
  fi
}

prepare_profiler_command() {
  local vendor="$1"
  local -n _cmd=$2
  local output_path="$3"

  local profiler_bin
  profiler_bin=$(resolve_profiler_binary "$vendor") || return 1

  case "$vendor" in
    nvidia)
      _cmd=(
        "$profiler_bin"
        -f
        -o "$output_path"
      )
      append_profiler_args "$vendor" _cmd
      return 0
      ;;
    intel)
      _cmd=(
        "$profiler_bin"
        -q --chrome-kernel-logging -g ComputeBasic
        -o "$output_path"
      )
      append_profiler_args "$vendor" _cmd
      return 0
      ;;
    amd)
      _cmd=("$profiler_bin")
      append_profiler_args "$vendor" _cmd
      return 2
      ;;
    *)
      return 1
      ;;
  esac
}

collect_metrics() {
  local vendor="${1,,}"
  local exe_path="$2"
  local json_path="$3"
  local dataset_name="$4"
  local label_classes="$5"
  local seed="$6"
  local metrics_dir="$7"
  local log_file="$8"

  if [[ -z "$vendor" ]]; then
    return 0
  fi

  local exe_name
  exe_name=$(resolve_exe_name "$exe_path")
  local output_path="${metrics_dir}/${exe_name}_${dataset_name}"

  local -a profiler_cmd
  local prep_status=0
  prepare_profiler_command "$vendor" profiler_cmd "$output_path" || prep_status=$?

  case "$prep_status" in
    0)
      ;;
    2)
      echo "Metric collection for $vendor GPUs is not implemented yet." >&2
      return 0
      ;;
    1)
      echo "Unsupported metrics vendor: $vendor" >&2
      return 1
      ;;
    *)
      return "$prep_status"
      ;;
  esac

  local profiler_bin="${profiler_cmd[0]}"
  if ! command -v "$profiler_bin" >/dev/null 2>&1; then
    echo "$profiler_bin command not found in PATH. Install the required tooling to collect metrics." >&2
    return 1
  fi

  local -a base_cmd
  build_run_command base_cmd "$exe_path" "$json_path" "$seed" "$label_classes" "$vendor"

  local output_suffix=""
  if [[ "$vendor" == "nvidia" ]]; then
    output_suffix=".ncu-rep"
  fi

  echo "Collecting ${vendor^^} metrics with $profiler_bin -> ${output_path}${output_suffix}" | tee -a "$log_file"
  "${profiler_cmd[@]}" "${base_cmd[@]}" >> "$log_file" 2>&1
}

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

run_experiment() {
  local exe_path="$1"
  local json_path="$2"
  local dataset_name="$3"
  local run_idx="$4"
  local seed="$5"
  local label_classes="$6"
  local log_file="$7"

  local exe_name
  exe_name=$(resolve_exe_name "$exe_path")
  local vendor
  vendor=$(resolve_exe_vendor "$exe_path")
  local -a base_cmd
  build_run_command base_cmd "$exe_path" "$json_path" "$seed" "$label_classes" "$vendor"

  echo "[${count}/${total}] Running $exe_name on $dataset_name (run $run_idx/$RUNS, labels $label_classes, vendor $vendor)" | tee -a "$log_file"

  "${base_cmd[@]}" >> "$log_file" 2>&1
}

echo "Running benchmarks for ${#EXECUTABLES[@]} implementations on ${#JSON_FILES[@]} datasets across ${#LABEL_CLASSES_LIST[@]} label-class settings"

total=$(( ${#EXECUTABLES[@]} * ${#JSON_FILES[@]} * ${#LABEL_CLASSES_LIST[@]} * RUNS ))
count=0

for exe_path in "${EXECUTABLES[@]}"; do
  exe_name=$(resolve_exe_name "$exe_path")
  for json_path in "${JSON_FILES[@]}"; do
    dataset_name=$(basename "$json_path" .json)

    for label_classes in "${LABEL_CLASSES_LIST[@]}"; do
      log_file="$LOG_DIR/${exe_name}_${dataset_name}_seed${LABEL_SEED}_labels${label_classes}.log"
      : > "$log_file"

      for run_idx in $(seq 1 "$RUNS"); do
        count=$((count + 1))
        if [ "$RUN_EXPERIMENT" = true ]; then
          run_experiment "$exe_path" "$json_path" "$dataset_name" "$run_idx" "$LABEL_SEED" "$label_classes" "$log_file"
        fi
      done

      if [[ -n "$COLLECT_METRICS" ]]; then
        collect_metrics "$COLLECT_METRICS" "$exe_path" "$json_path" "$dataset_name" "$label_classes" "$LABEL_SEED" "$METRICS_DIR" "$log_file"
      fi
    done
  done

done

echo "Benchmarking completed. Logs saved under $LOG_DIR"
