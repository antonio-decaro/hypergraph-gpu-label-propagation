cmake_minimum_required(VERSION 3.16)
project(HypergraphLabelPropagation 
        VERSION 1.0.0 
        LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add include directory
include_directories(include)

# Build common library
add_library(hypergraph_common STATIC
    src/common/hypergraph.cpp
)

target_include_directories(hypergraph_common PUBLIC include)

# Find and configure different programming models
set(BUILD_OPENMP ON CACHE BOOL "Build OpenMP implementation")
set(BUILD_SYCL OFF CACHE BOOL "Build SYCL implementation")  
set(BUILD_KOKKOS OFF CACHE BOOL "Build Kokkos implementation")

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
            OpenMP::OpenMP_CXX
        )
        
        target_include_directories(label_propagation_openmp PRIVATE src/openmp)
        
        # Set OpenMP compiler flags
        target_compile_options(label_propagation_openmp PRIVATE ${OpenMP_CXX_FLAGS})
    else()
        message(WARNING "OpenMP not found, skipping OpenMP implementation")
    endif()
endif()

# Build SYCL implementation
if(BUILD_SYCL)
    # Try to find SYCL compiler
    find_program(SYCL_COMPILER 
        NAMES icpx dpcpp sycl++
        DOC "SYCL compiler executable"
    )
    
    if(SYCL_COMPILER)
        message(STATUS "Building SYCL implementation with ${SYCL_COMPILER}")
        
        add_executable(label_propagation_sycl
            src/sycl/main_sycl.cpp
            src/sycl/label_propagation_sycl.cpp
        )
        
        target_link_libraries(label_propagation_sycl 
            PRIVATE 
            hypergraph_common
        )
        
        target_include_directories(label_propagation_sycl PRIVATE src/sycl)
        
        # Set SYCL compiler and flags
        set_target_properties(label_propagation_sycl PROPERTIES
            CXX_COMPILER ${SYCL_COMPILER}
        )
        
        target_compile_options(label_propagation_sycl PRIVATE 
            -fsycl
            -fsycl-targets=spir64,nvptx64-nvidia-cuda
        )
        
        target_link_options(label_propagation_sycl PRIVATE 
            -fsycl
            -fsycl-targets=spir64,nvptx64-nvidia-cuda
        )
    else()
        message(WARNING "SYCL compiler not found, skipping SYCL implementation")
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
message(STATUS "  OpenMP: ${BUILD_OPENMP}")
message(STATUS "  SYCL:   ${BUILD_SYCL}")  
message(STATUS "  Kokkos: ${BUILD_KOKKOS}")
message(STATUS "")

# Add custom target to build all available implementations
add_custom_target(all_implementations)

if(TARGET label_propagation_openmp)
    add_dependencies(all_implementations label_propagation_openmp)
endif()

if(TARGET label_propagation_sycl)
    add_dependencies(all_implementations label_propagation_sycl)
endif()

if(TARGET label_propagation_kokkos)
    add_dependencies(all_implementations label_propagation_kokkos)
endif()

# Installation
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path" FORCE)
endif()

if(TARGET label_propagation_openmp)
    install(TARGETS label_propagation_openmp DESTINATION bin)
endif()

if(TARGET label_propagation_sycl)
    install(TARGETS label_propagation_sycl DESTINATION bin)
endif()

if(TARGET label_propagation_kokkos)
    install(TARGETS label_propagation_kokkos DESTINATION bin)
endif()

install(FILES include/hypergraph.hpp DESTINATION include)

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