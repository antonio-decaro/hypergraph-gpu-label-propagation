#!/usr/bin/env bash
#PBS -N HLP 
#PBS -A 
#PBS -q debug
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=01:00:00
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
DEFAULT_TARGET_VENDOR="nvidia"
DATASET_FILTER=()
ALGORITHM_FILTER="all"  # all | lp | pr

# Target vendor configuration for workgroup sizing

# Map implementation names (from resolve_exe_name) to vendors when they differ from the default.
# Example: EXEC_VENDOR_OVERRIDES[openmp]="intel"
declare -A EXEC_VENDOR_OVERRIDES=()

declare -A PROFILER_BINARIES=(
  [nvidia]="ncu"
  [amd]="rocprof"
  [intel]="vtune"
)
declare -A PROFILER_ARGS=(
  [intel]="-collect gpu-hotspots -knob characterization-mode=overview"
)

# Workgroup sizes per vendor
declare -A WORKGROUP_SIZES=(
  [nvidia]=256
  [amd]=512
  [intel]=256
)

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --datasets NAME[,NAME,...]            Comma-separated list of dataset names to run (default: all)
  --algorithm {lp|pr|all}              Run only label propagation, only PageRank, or both (default: all)
  --collect-metrics {nvidia|amd|intel}  Collect GPU metrics (default: disabled)
  --metrics-dir PATH                    Directory for profiler outputs (default: LOG_DIR/metrics)
  --skip-run                            Skip executing the benchmark binaries; only collect metrics
  -h, --help                            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      IFS=',' read -r -a DATASET_FILTER <<< "${2:-}"
      shift 2
      ;;
    --algorithm)
      ALGORITHM_FILTER="${2:-all}"
      shift 2
      ;;
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
    label_propagation_sycl*)   echo "lp_sycl" ;;
    label_propagation_openmp*) echo "lp_openmp" ;;
    label_propagation_kokkos*) echo "lp_kokkos" ;;
    label_propagation_cuda*)   echo "lp_cuda" ;;
    page_rank_sycl*)           echo "pr_sycl" ;;
    page_rank_openmp*)         echo "pr_openmp" ;;
    page_rank_kokkos*)         echo "pr_kokkos" ;;
    page_rank_cuda*)           echo "pr_cuda" ;;
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
        --set full
        -f
        -o "$output_path"
      )
      append_profiler_args "$vendor" _cmd
      return 0
      ;;
    amd|intel)
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

  local -a base_cmd
  build_run_command base_cmd "$exe_path" "$json_path" "$seed" "$label_classes" "$vendor"

  echo "Collecting ${vendor^^} metrics -> ${output_path}" 

  if [[ "$vendor" == "intel" ]]; then
    vtune -collect gpu-hotspots -knob characterization-mode=overview -r "${output_path}_compute" -- "${base_cmd[@]}" 
    # vtune -collect gpu-hotspots -knob characterization-mode=global-memory-accesses -r "${output_path}_memory" -- "${base_cmd[@]}"
  elif [[ "$vendor" == "nvidia" ]]; then
    ncu --set full -f -o "${output_path}" "${base_cmd[@]}" 
  else
    echo "Unsupported vendor for metrics collection: $vendor" >&2
  fi
  
}

readarray -t JSON_FILES < <(find "$JSON_DIR" -maxdepth 1 -type f -name '*.json' | sort)
if [ ${#JSON_FILES[@]} -eq 0 ]; then
  echo "No JSON files found in $JSON_DIR" >&2
  exit 1
fi

if [ ${#DATASET_FILTER[@]} -gt 0 ]; then
  filtered=()
  for f in "${JSON_FILES[@]}"; do
    name=$(basename "$f" .json)
    for d in "${DATASET_FILTER[@]}"; do
      if [[ "$name" == "$d" ]]; then
        filtered+=("$f")
        break
      fi
    done
  done
  if [ ${#filtered[@]} -eq 0 ]; then
    echo "No matching datasets found for filter: ${DATASET_FILTER[*]}" >&2
    exit 1
  fi
  JSON_FILES=("${filtered[@]}")
fi

if [[ "$ALGORITHM_FILTER" != "lp" && "$ALGORITHM_FILTER" != "pr" && "$ALGORITHM_FILTER" != "all" ]]; then
  echo "Unknown --algorithm value: $ALGORITHM_FILTER (use lp, pr, or all)" >&2
  exit 1
fi

EXECUTABLES=()
while IFS= read -r exe; do
  case "$ALGORITHM_FILTER" in
    lp)  [[ "$exe" == *label_propagation_* ]] && EXECUTABLES+=("$exe") || true ;;
    pr)  [[ "$exe" == *page_rank_* ]]         && EXECUTABLES+=("$exe") || true ;;
    all) EXECUTABLES+=("$exe") ;;
  esac
done < <(find "$BUILD_DIR" -maxdepth 2 -type f -executable \( -name 'label_propagation_*' -o -name 'page_rank_*' \) | sort)

if [ ${#EXECUTABLES[@]} -eq 0 ]; then
  echo "No executables found in $BUILD_DIR for --algorithm=$ALGORITHM_FILTER" >&2
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

  if ! "${base_cmd[@]}" >> "$log_file" 2>&1; then
    local rc=$?
    echo "WARNING: $exe_name failed on $dataset_name run $run_idx (exit $rc)" | tee -a "$log_file" >&2
  fi
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
