# Hypergraph GPU Label Propagation

Label Propagation for hypergraphs across multiple backends: OpenMP (CPU and offload), SYCL, and Kokkos. A unified CLI configures generators, algorithm limits, labels, and binary I/O.

## Overview

- Implementations: OpenMP, SYCL, and Kokkos.
- Unified CLI via cxxopts for all backends.
- Built‑in random hypergraph generators: uniform, fixed (a.k.a. d‑uniform ER), planted partition, and hSBM.
- Binary save/load for reproducible experiments.

## Project Layout

```
include/                # Public headers (Hypergraph, generators, CLI options)
src/common/             # Shared hypergraph + generators + CLI parsing
src/openmp/             # OpenMP implementation + entry point
src/sycl/               # SYCL implementation + entry point
src/kokkos/             # Kokkos implementation + entry point
```

## Build

Prereqs
- CMake ≥ 3.16
- C++17 compiler
- Optional: SYCL toolchain; Kokkos installation

Configure + build (OpenMP)
```bash
mkdir build && cd build
cmake -DBUILD_OPENMP=ON ..
cmake --build . -j
```

Other backends
- SYCL: `cmake -DBUILD_SYCL=ON -DSYCL_TARGETS="spir64,nvptx64-nvidia-cuda" .. && cmake --build . -j`
- Kokkos: `cmake -DBUILD_KOKKOS=ON -DKokkos_ROOT=/path/to/kokkos .. && cmake --build . -j`

Install
```bash
cmake --install . --prefix ./install
```

## Unified CLI

General
- `--vertices, -v` number of vertices (default 1000)
- `--edges, -e` number of hyperedges (default 5000)
- `--iterations, -i` max iterations (default 100)
- `--tolerance, -t` convergence tolerance (default 1e-6)
- `--threads, -p` OpenMP threads (0=auto)
- `--seed` RNG seed for graph (0=nondeterministic)

Labels
- `--label-classes` number of classes to assign randomly (0=skip)
- `--label-seed` RNG seed for labels (0=nondeterministic)

I/O
- `--load <file>` load a hypergraph from binary
- `--save <file>` save the generated/loaded hypergraph to binary

Generator selection
- `--generator {uniform|fixed|planted|hsbm}` or shortcuts `--uniform`, `--fixed`, `--planted`, `--hsbm`

Generator parameters
- uniform: `--min-edge-size`, `--max-edge-size`
- fixed: `--edge-size` (d‑uniform ER G^(d)(n,m) is equivalent to fixed with edge‑size=d)
- planted: `--communities`, `--p-intra`, `--min-edge-size`, `--max-edge-size`
- hsbm: `--communities`, `--p-intra`, `--p-inter`, `--min-edge-size`, `--max-edge-size`

Help/Version
- `--help`, `--version`

Examples
```bash
# OpenMP: uniform generator
./label_propagation_openmp \
  --uniform -v 1000 -e 5000 --min-edge-size 2 --max-edge-size 6 \
  --iterations 100 --tolerance 1e-6 --threads 0 --save data/uniform.bin

# Fixed (aka d-uniform ER with d=4)
./label_propagation_openmp --fixed -v 2000 -e 10000 --edge-size 4

# Planted partition
./label_propagation_openmp --planted -v 4000 -e 20000 \
  --communities 8 --p-intra 0.85 --min-edge-size 2 --max-edge-size 5

# hSBM with explicit inter probability
./label_propagation_openmp --hsbm -v 5000 -e 20000 \
  --communities 10 --p-intra 0.9 --p-inter 0.05 --min-edge-size 3 --max-edge-size 6

# Load from file and re-label
./label_propagation_openmp --load data/uniform.bin --label-classes 6 --label-seed 42
```

## Generators (brief)
- uniform: Edge sizes sampled uniformly in `[min-edge-size, max-edge-size]`; vertices chosen uniformly without replacement.
- fixed: All edges have exactly `edge-size` vertices. Equivalent to d‑uniform ER when you view edges as uniform d‑subsets.
- planted: Partition vertices into `communities`. For each edge, sample size in range; with probability `p-intra`, draw mostly from a single community (fill from outside if needed); otherwise mix across communities.
- hSBM: Rejection sampling. For a random k‑set, accept with `p-intra` if all vertices are in the same community; otherwise with `p-inter`. Edge size k sampled in range.

## Binary format
Saved via `--save` and loaded via `--load` (little‑endian):
- uint32 magic: `HGR1`
- uint32 version: `1`
- uint64 `num_vertices`
- uint64 `num_edges`
- For each edge: uint64 `edge_size`, followed by `edge_size` x uint64 vertex ids
- uint8 `has_labels`
- If `has_labels`: `num_vertices` x int32 labels

## Algorithm (Label Propagation)
- Per iteration, each vertex adopts the label maximizing the weighted count from incident hyperedges (weight 1/edge_size per neighbor occurrence).
- Stop when the fraction of label changes drops below tolerance or after `--iterations`.

## Notes
- OpenMP target offload path uses a flattened hypergraph representation for device access.
- `benchmark.sh` contains simple driver samples for quick runs.

## License
Apache 2.0. See LICENSE.
