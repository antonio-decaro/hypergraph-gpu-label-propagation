cmake_minimum_required(VERSION 3.16)
project(HypergraphLabelPropagation 
        VERSION 1.0.0 
        LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global include directory for public headers
include_directories(include)

# Resolve cxxopts via bundled module
include(cmake/FetchCXXOPTS.cmake)

# Build common library
add_library(hypergraph_common STATIC
    src/common/hypergraph.cpp
)

target_include_directories(hypergraph_common PUBLIC include)

# CLI common (arg parsing + hypergraph construction)
add_library(cli_common STATIC
    src/common/argparse.cpp
)
target_include_directories(cli_common PUBLIC include)
if(DEFINED CXXOPTS_IMPORTED_INCLUDE)
    target_include_directories(cli_common PRIVATE ${CXXOPTS_IMPORTED_INCLUDE})
else()
    message(WARNING "cxxopts not available. Set CXXOPTS_INCLUDE_DIR or enable DOWNLOAD_CXXOPTS.")
endif()

# Find and configure different programming models
set(BUILD_OPENMP OFF CACHE BOOL "Build OpenMP implementation")
set(BUILD_SYCL OFF CACHE BOOL "Build SYCL implementation")  
set(BUILD_KOKKOS OFF CACHE BOOL "Build Kokkos implementation")

set(OFFLOAD_VENDOR "NVIDIA" CACHE STRING "Offload target vendor: NVIDIA, AMD, or INTEL")
set_property(CACHE OFFLOAD_VENDOR PROPERTY STRINGS "NVIDIA" "AMD" "INTEL")

set(OFFLOAD_TARGET "none" CACHE STRING "Offload target architecture (none, sm_70, sm_80, sm_86, gfx908, gfx90a, x86_64, etc.)")

# OpenMP target offload configuration (module)
include(cmake/OpenMPOffload.cmake)
include(cmake/SYCLOffload.cmake)

# Build OpenMP implementation
if(BUILD_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Building OpenMP implementation")
        
        add_executable(label_propagation_openmp
            src/openmp/main_openmp.cpp
            src/openmp/label_propagation_openmp.cpp
        )
        
        target_link_libraries(label_propagation_openmp 
            PRIVATE 
            hypergraph_common
            cli_common
            OpenMP::OpenMP_CXX
        )
        
        target_include_directories(label_propagation_openmp PRIVATE src/openmp)
        
        # Set OpenMP compiler flags
        target_compile_options(label_propagation_openmp PRIVATE ${OpenMP_CXX_FLAGS})

        # Configure OpenMP target offload via helper
        omp_configure_offload(label_propagation_openmp)
    else()
        message(WARNING "OpenMP not found, skipping OpenMP implementation")
    endif()
endif()

# Build SYCL implementation
if(BUILD_SYCL)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fsycl" CMAKE_CXX_COMPILER_SUPPORTS_FSYCL)

    if(CMAKE_CXX_COMPILER_SUPPORTS_FSYCL)
        set(BUILD_SYCL ON CACHE BOOL "Build SYCL implementation" FORCE)
        message(STATUS "Building SYCL implementation with ${CMAKE_CXX_COMPILER}")

        add_executable(label_propagation_sycl
            src/sycl/main_sycl.cpp
            src/sycl/label_propagation_sycl.cpp
        )

        target_link_libraries(label_propagation_sycl 
            PRIVATE 
            hypergraph_common
            cli_common
        )

        target_include_directories(label_propagation_sycl PRIVATE src/sycl)

        sycl_configure_offload(label_propagation_sycl)
    else()
        message(WARNING "Selected compiler (${CMAKE_CXX_COMPILER}) does not support -fsycl. Skipping SYCL implementation.")
        set(BUILD_SYCL OFF CACHE BOOL "Build SYCL implementation" FORCE)
    endif()
endif()

# Build Kokkos implementation
if(BUILD_KOKKOS)
    find_package(Kokkos QUIET)
    
    if(Kokkos_FOUND)
        message(STATUS "Building Kokkos implementation")
        
        add_executable(label_propagation_kokkos
            src/kokkos/main_kokkos.cpp
            src/kokkos/label_propagation_kokkos.cpp
        )
        
        target_link_libraries(label_propagation_kokkos 
            PRIVATE 
            hypergraph_common
            cli_common
            Kokkos::kokkos
        )
        
        target_include_directories(label_propagation_kokkos PRIVATE src/kokkos)
    else()
        message(WARNING "Kokkos not found, skipping Kokkos implementation")
        message(STATUS "To build Kokkos implementation:")
        message(STATUS "  1. Install Kokkos library")
        message(STATUS "  2. Set CMAKE_PREFIX_PATH or Kokkos_ROOT")
        message(STATUS "  3. Set -DBUILD_KOKKOS=ON")
    endif()
endif()

# Print build summary
message(STATUS "")
message(STATUS "Build Configuration Summary:")
message(STATUS "  OpenMP: ${BUILD_OPENMP} | OMP Offload=${ENABLE_OPENMP_OFFLOAD} arch='${OMP_TARGET_ARCH}'")
message(STATUS "  SYCL:   ${BUILD_SYCL}")  
message(STATUS "  Kokkos: ${BUILD_KOKKOS}")
message(STATUS "")

# Add usage information
message(STATUS "Usage:")
message(STATUS "  Build all available: make all_implementations")
message(STATUS "  Build specific:")
if(TARGET label_propagation_openmp)
    message(STATUS "    OpenMP: make label_propagation_openmp")
endif()
if(TARGET label_propagation_sycl)
    message(STATUS "    SYCL:   make label_propagation_sycl")
endif()
if(TARGET label_propagation_kokkos)
    message(STATUS "    Kokkos: make label_propagation_kokkos") 
endif()
message(STATUS "  Install: make install")
