# OpenMP implementation
cmake_minimum_required(VERSION 3.16)

find_package(OpenMP REQUIRED)

add_executable(label_propagation_openmp
    main_openmp.cpp
    label_propagation_openmp.cpp
)

target_link_libraries(label_propagation_openmp 
    PRIVATE 
    hypergraph_common
    OpenMP::OpenMP_CXX
)

target_include_directories(label_propagation_openmp PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    .
)

# Set OpenMP compiler flags
target_compile_options(label_propagation_openmp PRIVATE ${OpenMP_CXX_FLAGS})