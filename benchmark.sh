#!/bin/bash

# Benchmark script for hypergraph label propagation implementations
# Usage: ./benchmark.sh [vertices] [edges] [iterations]

set -e

VERTICES=${1:-1000}
EDGES=${2:-5000}
ITERATIONS=${3:-100}

echo "Hypergraph Label Propagation Benchmark"
echo "======================================"
echo "Parameters: $VERTICES vertices, $EDGES edges, max $ITERATIONS iterations"
echo

BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Please run:"
    echo "  mkdir build && cd build"
    echo "  cmake -DBUILD_OPENMP=ON -DBUILD_SYCL=ON -DBUILD_KOKKOS=ON .."
    echo "  make all_implementations"
    exit 1
fi

cd "$BUILD_DIR"

# Test OpenMP implementation
if [ -f "label_propagation_openmp" ]; then
    echo "Testing OpenMP implementation:"
    echo "------------------------------"
    for threads in 1 2 4 8; do
        if [ $threads -le $(nproc) ]; then
            echo "OpenMP with $threads threads:"
            ./label_propagation_openmp $VERTICES $EDGES $ITERATIONS $threads | grep -E "(Runtime:|Iterations:)"
            echo
        fi
    done
else
    echo "OpenMP implementation not found"
fi

# Test SYCL implementation
if [ -f "label_propagation_sycl" ]; then
    echo "Testing SYCL implementation:"
    echo "----------------------------"
    for device in auto cpu gpu; do
        echo "SYCL with $device device:"
        ./label_propagation_sycl $VERTICES $EDGES $ITERATIONS $device 2>/dev/null | grep -E "(Runtime:|Iterations:)" || echo "Failed to run with $device device"
        echo
    done
else
    echo "SYCL implementation not found"
fi

# Test Kokkos implementation
if [ -f "label_propagation_kokkos" ]; then
    echo "Testing Kokkos implementation:"
    echo "------------------------------"
    ./label_propagation_kokkos $VERTICES $EDGES $ITERATIONS | grep -E "(Runtime:|Iterations:)"
    echo
else
    echo "Kokkos implementation not found"
fi

echo "Benchmark completed!"
echo
echo "To build missing implementations:"
echo "- OpenMP: Already available on most systems"
echo "- SYCL: Install Intel oneAPI DPC++ toolkit"
echo "- Kokkos: Install from https://github.com/kokkos/kokkos"