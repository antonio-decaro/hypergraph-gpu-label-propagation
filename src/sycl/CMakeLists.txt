# SYCL implementation
cmake_minimum_required(VERSION 3.16)

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

target_compile_options(label_propagation_sycl PRIVATE 
    -fsycl
    -fsycl-targets=spir64,nvptx64-nvidia-cuda
)

target_link_options(label_propagation_sycl PRIVATE 
    -fsycl
    -fsycl-targets=spir64,nvptx64-nvidia-cuda
)