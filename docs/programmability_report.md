# Hypergraph Label Propagation – Programmability Report

This document compares the three accelerator back ends currently in the repository (SYCL, OpenMP target offload, Kokkos) with emphasis on programmability and the code volume required to implement the label propagation algorithm after the latest refactors.

## TL;DR

- **OpenMP target** remains the least verbose (∼280 LOC for the core translation unit) and is quick to prototype but relies on careful `map` management and lacks explicit subgroup control.
- **SYCL** offers the most concise expression of hierarchical parallelism (∼160 LOC in the core header) and first-class subgroup/local memory support, at the cost of explicit queue/USM management.
- **Kokkos** is now the most feature-rich front end (∼530 LOC) with explicit execution-pool tiers (team, vector, scalar) and portable scratch usage; verbosity grows with templates and manual pool plumbing, but it gives the broadest backend coverage.

## Code Size Snapshot

Measured with `wc -l` on 2024-XX-XX commit state; counts include implementation files only (not headers other than SYCL’s single header).

| Model  | Core Files                                                                 | LOC |
|--------|-----------------------------------------------------------------------------|-----|
| SYCL   | `src/sycl/label_propagation_sycl.hpp`                                       | 157 |
| OpenMP | `src/openmp/label_propagation_openmp.cpp`                                   | 282 |
| Kokkos | `src/kokkos/label_propagation_kokkos.cpp`                                   | 527 |

The main entry points (`src/*/main_*.cpp`) add between 45–60 LOC each and were not counted in the table. SLOC (non-comment, non-blank) roughly tracks the same ordering: SYCL < OpenMP << Kokkos.

## Kernel & Memory Model Comparison

- **SYCL (`src/sycl/label_propagation_sycl.hpp`)**
  - Two `parallel_for` kernels per iteration launched as `nd_range`.
  - Uses USM to keep vertex/edge arrays contiguous; `local_accessor` provides scratchpad memory per workgroup.
  - Sub-group operations available; current code relies on workgroup-level synchronization.
  - Data movement is explicit (queue copies) but deterministic.

- **OpenMP Target (`src/openmp/label_propagation_openmp.cpp`)**
  - Execution pools split edges/vertices between workgroup-style teams and scalar work-items.
  - `target data` region keeps the flattened hypergraph resident; each iteration runs `target teams` kernels with reductions.
  - Memory management is implicit via `map` clauses; correctness hinges on matching `present` arguments and reduction semantics.
  - No native subgroup abstraction, so cooperative work relies on nested `omp parallel` regions and atomics or reduction clauses.

- **Kokkos (`src/kokkos/label_propagation_kokkos.cpp`)**
  - Pools now have three tiers: full team (`TeamPolicy` with scratch), vector (`TeamPolicy` with `ThreadVectorRange`), and scalar (`RangePolicy`).
  - Device data stored in `View`s with host mirrors; scratch memory allocated per team for label histograms.
  - Explicit control of team size, vector length, and scratch footprint; reductions performed with `parallel_reduce`.
  - Additional boilerplate for execution pool creation and mirror copies, but portable across CUDA/HIP/OpenMP back ends.

## Programmability Discussion

| Aspect              | SYCL                                             | OpenMP Target                                     | Kokkos                                                    |
|---------------------|--------------------------------------------------|---------------------------------------------------|-----------------------------------------------------------|
| Hierarchical control| `nd_range`, sub-groups, local memory             | Teams/threads only; no subgroup primitive         | Team / thread / vector tiers, configurable scratch        |
| Memory management   | Explicit USM allocations, queue copies           | Host vectors with `map` clauses                   | `View`/mirror pairs, `deep_copy`, scratch allocations     |
| Data residency      | Manual but fine-grained                          | `target data` keeps arrays on device automatically| Managed through Views; persistent across kernels          |
| Tuning knobs        | Workgroup size, local memory, subgroup width     | `num_teams`, `thread_limit`, clause tuning        | Team size, vector length, scratch size, execution space   |
| Tooling & portability| oneAPI/hipSYCL/ComputeCpp (CPU, GPU, FPGA)      | Depends on compiler offload support (Clang, GCC)  | Single-source across CUDA, HIP, SYCL*, OpenMP, Serial     |
| Code verbosity      | Low                                              | Moderate                                          | High (due to templates and explicit pools)                |

## Maintenance Notes

- The custom execution pool logic dominates all three implementations. Factoring pool creation and iteration scaffolding into shared utilities could reduce duplication and LOC in every backend.
- SYCL and Kokkos now share the same conceptual tiers (workgroup/team, subgroup/vector, work-item/scalar). Maintaining equivalent behavior requires keeping shared constants (e.g., subgroup size, label cap) synchronized.
- OpenMP relies on pragmatic caching (`target data`), but the `present` clauses must be kept consistent as the data structures evolve; regression tests that exercise empty pools and small hypergraphs are recommended.

## File References

- SYCL backend: `src/sycl/label_propagation_sycl.hpp`
- OpenMP backend: `src/openmp/label_propagation_openmp.cpp`
- Kokkos backend: `src/kokkos/label_propagation_kokkos.cpp`

For entry points and CLI wiring, refer to `src/sycl/main_sycl.cpp`, `src/openmp/main_openmp.cpp`, and `src/kokkos/main_kokkos.cpp`.
