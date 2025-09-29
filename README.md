# Hypergraph GPU Label Propagation

Implementation of the Label Propagation Algorithm for Hypergraphs on GPUs using different heterogeneous programming models: **OpenMP**, **SYCL**, and **Kokkos**.

## Overview

This project implements the label propagation algorithm for hypergraphs using three different parallel programming models:

- **OpenMP**: CPU-based parallel implementation using OpenMP threading
- **SYCL**: GPU/accelerator implementation using SYCL (Intel DPC++, hipSYCL, etc.)
- **Kokkos**: Performance-portable implementation supporting multiple backends (CUDA, HIP, OpenMP, etc.)

## Project Structure

```
├── include/
│   └── hypergraph.hpp          # Common hypergraph data structures and interfaces
├── src/
│   ├── common/
│   │   └── hypergraph.cpp      # Common hypergraph implementation
│   ├── openmp/
│   │   ├── label_propagation_openmp.hpp
│   │   ├── label_propagation_openmp.cpp
│   │   ├── main_openmp.cpp
│   │   └── CMakeLists.txt
│   ├── sycl/
│   │   ├── label_propagation_sycl.hpp
│   │   ├── label_propagation_sycl.cpp
│   │   ├── main_sycl.cpp
│   │   └── CMakeLists.txt
│   └── kokkos/
│       ├── label_propagation_kokkos.hpp
│       ├── label_propagation_kokkos.cpp
│       ├── main_kokkos.cpp
│       └── CMakeLists.txt
└── CMakeLists.txt              # Main build configuration
```

## Requirements

### General Requirements
- CMake 3.16 or later
- C++17 compatible compiler

### OpenMP Implementation
- OpenMP-capable compiler (GCC, Clang, Intel, etc.)

### SYCL Implementation
- SYCL-compatible compiler:
  - Intel DPC++ (oneAPI toolkit)
  - hipSYCL
  - ComputeCpp
  - triSYCL (CPU-only)

### Kokkos Implementation
- Kokkos library installed (3.0 or later)
- Compatible backend (CUDA, HIP, OpenMP, etc.)

## Building

### Quick Start (OpenMP only)

```bash
mkdir build && cd build
cmake -DBUILD_OPENMP=ON ..
make label_propagation_openmp
```

### Building All Available Implementations

```bash
mkdir build && cd build
cmake -DBUILD_OPENMP=ON -DBUILD_SYCL=ON -DBUILD_KOKKOS=ON ..
make all_implementations
```

### Building Specific Implementations

#### OpenMP Implementation
```bash
mkdir build && cd build
cmake -DBUILD_OPENMP=ON ..
make label_propagation_openmp
```

#### SYCL Implementation
```bash
# Ensure SYCL compiler is in PATH
mkdir build && cd build
cmake -DBUILD_SYCL=ON ..
make label_propagation_sycl
```

#### Kokkos Implementation
```bash
# Ensure Kokkos is installed and findable by CMake
mkdir build && cd build
cmake -DBUILD_KOKKOS=ON -DKokkos_ROOT=/path/to/kokkos ..
make label_propagation_kokkos
```

### CMake Configuration Options

- `BUILD_OPENMP` (default: ON): Build OpenMP implementation
- `BUILD_SYCL` (default: OFF): Build SYCL implementation
- `BUILD_KOKKOS` (default: OFF): Build Kokkos implementation

## Installation

```bash
make install
```

By default, binaries are installed to `build/install/bin/`. You can change the installation prefix:

```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
```

## Usage

All implementations accept the same command-line arguments:

```bash
./label_propagation_<implementation> [num_vertices] [num_edges] [max_iterations] [implementation_specific_args]
```

### Examples

```bash
# OpenMP with 1000 vertices, 5000 edges, max 100 iterations, 4 threads
./label_propagation_openmp 1000 5000 100 4

# SYCL with 1000 vertices, 5000 edges, max 100 iterations, GPU device
./label_propagation_sycl 1000 5000 100 gpu

# Kokkos with 1000 vertices, 5000 edges, max 100 iterations
./label_propagation_kokkos 1000 5000 100
```

## Algorithm Details

The label propagation algorithm for hypergraphs works as follows:

1. Initialize each vertex with a random label
2. In each iteration, update each vertex's label to the most frequent label among its neighbors
3. For hypergraphs, neighbors are defined through hyperedge connectivity
4. Weight contributions by edge size (1/edge_size)
5. Continue until convergence or maximum iterations reached

## Performance Considerations

- **OpenMP**: Best for CPU-bound workloads with good thread scalability
- **SYCL**: Optimized for GPU acceleration with automatic memory management
- **Kokkos**: Performance-portable across different architectures

## Dependencies Installation

### Installing SYCL (Intel oneAPI)

```bash
# Ubuntu/Debian
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-dpcpp-cpp
```

### Installing Kokkos

```bash
git clone https://github.com/kokkos/kokkos.git
cd kokkos
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON ..
make install
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your implementation or improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenMP Architecture Review Board for the OpenMP specification
- The Khronos Group for the SYCL specification
- Sandia National Laboratories for the Kokkos library
