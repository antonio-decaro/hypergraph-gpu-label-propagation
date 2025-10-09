#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

openmp_compiler=""
kokkos_compiler=""
sycl_compiler=""
cuda_compiler=""
arch=""
vendor=""

help()
{
  echo "Usage: ./build.sh 
    [ --sycl=compiler ] Path to the SYCL compiler (e.g., dpcpp, hipcc);
    [ --openmp=compiler ] Path to the OpenMP compiler (e.g., g++, clang++, icpc);
    [ --kokkos=compiler ] Path to the Kokkos compiler (e.g., g++, clang++, hipcc, nvcc);
    [ --cuda=compiler ] Path to the CUDA compiler (e.g., nvcc);
    [ --arch=architecture ] The target architecture (e.g., native, skylake, zen2, sm_70, gfx90a);
    [ -h | --help ] Print this help message and exit.
  "
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sycl=*)
      sycl_compiler="${1#*=}"
      shift
      ;;
    --openmp=*)
      openmp_compiler="${1#*=}"
      shift
      ;;
    --kokkos=*)
      kokkos_compiler="${1#*=}"
      shift
      ;;
    --cuda=*)
      cuda_compiler="${1#*=}"
      shift
      ;;
    --arch=*)
      arch="${1#*=}"
      shift
      ;;
    --vendor=*)
      vendor="${1#*=}"
      shift
      ;;
    --arch=*)
      arch="${1#*=}"
      shift
      ;;
    -h | --help)
      help
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      help
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

mkdir -p "$SCRIPT_DIR/build/"

# array to hold background process PIDs
declare -a pids=()
# associative array to map pid -> build name for clearer failure messages
declare -A pid_name_map=()

if [ -z "$sycl_compiler" ] && [ -z "$openmp_compiler" ] && [ -z "$kokkos_compiler" ] && [ -z "$cuda_compiler" ]; then
  echo "Error: At least one of --sycl, --openmp, --kokkos, or --cuda must be specified."
  help
  exit 1
fi

if [ -n "$cuda_compiler" ]; then
  vendor=${vendor:-NVIDIA}
fi

if [ -z "$arch" ] || { [ -z "$vendor" ] && [ -z "$cuda_compiler" ]; }; then
  echo "Error: --vendor and --arch must be specified."
  help
  exit 1
fi

# if sycl compiler is specified
if [[ -n "$sycl_compiler" ]]; then
  echo "Building with SYCL compiler: $sycl_compiler ..."
  (
    cmake -S $SCRIPT_DIR -B "$SCRIPT_DIR/build/sycl/" -DCMAKE_CXX_COMPILER="$sycl_compiler" -DCMAKE_BUILD_TYPE=Release -DBUILD_SYCL=ON -DOFFLOAD_VENDOR="$vendor" -DOFFLOAD_TARGET="$arch" > "$SCRIPT_DIR/build/cmake_sycl.log" 2>&1 && \
    cmake --build "$SCRIPT_DIR/build/sycl/" -t label_propagation_sycl -j 8 > "$SCRIPT_DIR/build/cmake_sycl_build.log" 2>&1 && \
    cp "$SCRIPT_DIR/build/sycl/label_propagation_sycl" "$SCRIPT_DIR/build/label_propagation_sycl"
  ) &
  pid=$!
  pids+=("$pid")
  pid_name_map["$pid"]="SYCL"
fi

# if openmp compiler is specified
if [[ -n "$openmp_compiler" ]]; then
  echo "Building with OpenMP compiler: $openmp_compiler ..."
  (
    cmake -S $SCRIPT_DIR -B "$SCRIPT_DIR/build/openmp/" -DCMAKE_CXX_COMPILER="$openmp_compiler" -DCMAKE_BUILD_TYPE=Release -DBUILD_OPENMP=ON -DOFFLOAD_VENDOR="$vendor" -DOFFLOAD_TARGET="$arch" > "$SCRIPT_DIR/build/cmake_openmp.log" 2>&1 && \
    cmake --build "$SCRIPT_DIR/build/openmp/" -t label_propagation_openmp -j 8 > "$SCRIPT_DIR/build/cmake_openmp_build.log" 2>&1 && \
    cp "$SCRIPT_DIR/build/openmp/label_propagation_openmp" "$SCRIPT_DIR/build/label_propagation_openmp"
  ) &
  pid=$!
  pids+=("$pid")
  pid_name_map["$pid"]="OpenMP"
fi

# if kokkos compiler is specified
if [[ -n "$kokkos_compiler" ]]; then
  echo "Building with Kokkos compiler: $kokkos_compiler ..."
  (
    cmake -S $SCRIPT_DIR -B "$SCRIPT_DIR/build/kokkos/" -DCMAKE_CXX_COMPILER="$kokkos_compiler" -DCMAKE_BUILD_TYPE=Release -DBUILD_KOKKOS=ON -DOFFLOAD_VENDOR="$vendor" -DOFFLOAD_TARGET="$arch" > "$SCRIPT_DIR/build/cmake_kokkos.log" 2>&1 && \
    cmake --build "$SCRIPT_DIR/build/kokkos/" -t label_propagation_kokkos -j 8 > "$SCRIPT_DIR/build/cmake_kokkos_build.log" 2>&1 && \
    cp "$SCRIPT_DIR/build/kokkos/label_propagation_kokkos" "$SCRIPT_DIR/build/label_propagation_kokkos"
  ) &
  pid=$!
  pids+=("$pid")
  pid_name_map["$pid"]="Kokkos"
fi

# if cuda compiler is specified
if [[ -n "$cuda_compiler" ]]; then
  echo "Building with CUDA compiler: $cuda_compiler ..."
  (
    cmake -S $SCRIPT_DIR -B "$SCRIPT_DIR/build/cuda/" -DCMAKE_CUDA_COMPILER="$cuda_compiler" -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON -DOFFLOAD_VENDOR="${vendor:-NVIDIA}" -DOFFLOAD_TARGET="$arch" > "$SCRIPT_DIR/build/cmake_cuda.log" 2>&1 && \
    cmake --build "$SCRIPT_DIR/build/cuda/" -t label_propagation_cuda -j 8 > "$SCRIPT_DIR/build/cmake_cuda_build.log" 2>&1 && \
    cp "$SCRIPT_DIR/build/cuda/label_propagation_cuda" "$SCRIPT_DIR/build/label_propagation_cuda"
  ) &
  pid=$!
  pids+=("$pid")
  pid_name_map["$pid"]="CUDA"
fi

# wait for background build processes to finish
if [ ${#pids[@]} -gt 0 ]; then
  echo "Waiting for ${#pids[@]} build process(es) to finish..."
  wait_status=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      : # successful
    else
      wait_status=1
      build_name="${pid_name_map[$pid]:-unknown}"
      echo "A build process (PID $pid) for '$build_name' failed." >&2
    fi
  done
  if [ $wait_status -ne 0 ]; then
    echo "One or more build processes failed." >&2
    exit 1
  else 
    echo "All build processes completed successfully."
  fi
fi
