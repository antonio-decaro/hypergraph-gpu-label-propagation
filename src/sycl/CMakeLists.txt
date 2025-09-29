# SYCL implementation
cmake_minimum_required(VERSION 3.16)

# Try to find SYCL compiler
find_program(SYCL_COMPILER 
    NAMES icpx dpcpp sycl++
    DOC "SYCL compiler executable"
)

if(NOT SYCL_COMPILER)
    message(FATAL_ERROR "SYCL compiler not found. Install Intel DPC++ or set SYCL compiler path")
endif()

add_executable(label_propagation_sycl
    main_sycl.cpp
    label_propagation_sycl.cpp
)

target_link_libraries(label_propagation_sycl 
    PRIVATE 
    hypergraph_common
)

target_include_directories(label_propagation_sycl PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    .
)

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