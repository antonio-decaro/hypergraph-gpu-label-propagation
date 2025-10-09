# Hypergraph Label Propagation: Programmability Analysis (SYCL, OpenMP, Kokkos)

This report compares programmability aspects across three implementations of the same algorithm (hypergraph label propagation): SYCL, OpenMP target offload, and Kokkos. It includes code size metrics, core-kernel structure, memory management patterns, tuning knobs, portability, and developer ergonomics.

## Executive Summary

- OpenMP target is the most compact for this workload; minimal boilerplate due to pragmas and direct use of host containers.
- SYCL offers explicit control of device memory (USM), local memory, and kernel launch configuration with good portability; code size sits between OpenMP and Kokkos.
- Kokkos is slightly more verbose than SYCL (especially with TeamPolicy + team scratch), but integrates deeply with C++ and provides backend portability (CUDA/HIP/OpenMP) via a single API.

## Code Size Metrics

Measured on current repo state; totals include `.hpp + .cpp` for the algorithm, excluding mains.

SLOC definition used here: counts non-blank, non-comment lines (blank lines and `//` comment-only lines are excluded). It serves as a rough proxy for code volume, not quality or performance.

- SYCL: 103 + 174 = 277 total lines; 208 SLOC
  - Files: `src/sycl/label_propagation_sycl.hpp`, `src/sycl/label_propagation_sycl.cpp`
  - Main: 61 total lines; 46 SLOC (`src/sycl/main_sycl.cpp:1`)
- OpenMP: 30 + 140 = 170 total lines; 141 SLOC
  - Files: `src/openmp/label_propagation_openmp.hpp`, `src/openmp/label_propagation_openmp.cpp`
  - Main: 48 total lines; 35 SLOC (`src/openmp/main_openmp.cpp:1`)
- Kokkos: 63 + 234 = 297 total lines; 221 SLOC
  - Files: `src/kokkos/label_propagation_kokkos.hpp`, `src/kokkos/label_propagation_kokkos.cpp`
  - Main: 52 total lines; 40 SLOC (`src/kokkos/main_kokkos.cpp:1`)

Approximate “kernel/core-logic” region sizes (iteration loop and kernels):
- SYCL core: ~80 SLOC in `run_iteration_sycl()` `src/sycl/label_propagation_sycl.cpp:1`
- OpenMP core: ~91 SLOC in iteration loop `src/openmp/label_propagation_openmp.cpp:1`
- Kokkos core: ~99 SLOC in iteration loop `src/kokkos/label_propagation_kokkos.cpp:1`

## Kernel Structure and Memory Patterns

### SYCL
- Device selection and queue management in `LabelPropagationSYCL` constructor.
- USM allocations for flattened hypergraph and label arrays.
- Two kernels per iteration (`nd_range` launches):
  - Edge update using `local_accessor` for per-work-item label counts; `src/sycl/label_propagation_sycl.cpp:1`.
  - Vertex update using `local_accessor` and a SYCL reduction over changes; `src/sycl/label_propagation_sycl.cpp:1`.
- Pros: precise control of launch parameters and local memory; portable across backends; standard C++ with SYCL extensions.
- Cons: explicit memory management and queue orchestration adds boilerplate; learning curve for accessors/reductions.

### OpenMP Target
- Uses host vectors directly with `map` clauses to move data.
- Two offload regions per iteration with `target teams distribute parallel for` and a reduction for changes; `src/openmp/label_propagation_openmp.cpp:1`.
- Pros: minimal code changes from CPU version; very concise; familiar pragmas for many HPC devs; easy incremental adoption.
- Cons: portability depends on compiler offload maturity; fine-grained control (e.g., shared memory, groups) is less explicit; tricky `map` tuning for performance.

### Kokkos
- Flattens hypergraph into Kokkos `View`s with host mirrors and `deep_copy`.
- Two phases per iteration implemented with `TeamPolicy`:
  - Edge update: per-team scratch memory (shared) sized to `MAX_LABELS * team_size`, each thread uses a slice for counts; `src/kokkos/label_propagation_kokkos.cpp:1`.
  - Vertex update: same scratch usage and a `parallel_reduce` over `TeamThreadRange` to accumulate changes; `src/kokkos/label_propagation_kokkos.cpp:1`.
- Pros: Single-source performance-portable C++; explicit control over teams, scratch (shared) memory; integrates well with C++ tooling.
- Cons: Slightly more verbose than SYCL; mental model of `TeamPolicy`, scratch sizing, and mirrors/deep_copy required.

## Tuning Knobs and Portability

- SYCL
  - Workgroup size: `--workgroup-size` controls `nd_range` local size.
  - Local memory use mirrors CUDA shared mem via `local_accessor`.
  - Backends: Level Zero, OpenCL, CUDA, HIP (depends on toolchain).

- OpenMP
  - Teams/threads: controlled indirectly by environment variables and pragmas; mapping clauses heavily influence performance.
  - Portability: relies on compiler’s target offload support (Clang, GCC, vendor compilers).

- Kokkos
  - Team size: derived from `--workgroup-size`; backend chooses vector length.
  - Scratch memory: set via `policy.set_scratch_size(PerTeam(...))`.
  - Backends: CUDA, HIP, SYCL (experimental in some versions), OpenMP, Serial.

## Developer Ergonomics

- Build complexity: SYCL and Kokkos need additional toolchains or libraries; OpenMP often works with system compilers that support target offload.
- Memory management: OpenMP abstracts transfers via `map`; SYCL/Kokkos explicitly manage device memory (USM/Views) but yield more control.
- Debugging: OpenMP pragmas are easy to disable; SYCL has tools like `sycl-ls`, profilers from vendors; Kokkos has `Kokkos::Tools` and backend profilers.
- Maintenance: OpenMP is shortest; SYCL/Kokkos provide clearer performance tuning knobs and explicit data movement which can scale better as complexity grows.

## Algorithm Parity

All backends implement the same two-phase label propagation with a bounded label space (`MAX_LABELS=10`) and majority voting on incident elements. The SYCL and Kokkos versions leverage on-chip memory (local/scratch) to hold per-thread label counts, matching the same optimization.

## Recommendations

- If lowest development effort is key and compilers support target well: OpenMP target is attractive.
- For cross-vendor, modern C++ with explicit control and good ecosystem: SYCL is a solid choice.
- For HPC C++ projects seeking long-term performance portability across accelerators with single API: Kokkos provides a strong balance, at modest extra verbosity.

## File References

- SYCL implementation entry points: `src/sycl/label_propagation_sycl.hpp:1`, `src/sycl/label_propagation_sycl.cpp:1`, `src/sycl/main_sycl.cpp:1`
- OpenMP implementation entry points: `src/openmp/label_propagation_openmp.hpp:1`, `src/openmp/label_propagation_openmp.cpp:1`, `src/openmp/main_openmp.cpp:1`
- Kokkos implementation entry points: `src/kokkos/label_propagation_kokkos.hpp:1`, `src/kokkos/label_propagation_kokkos.cpp:1`, `src/kokkos/main_kokkos.cpp:1`
