# Kokkos implementation
cmake_minimum_required(VERSION 3.16)

find_package(Kokkos REQUIRED)

add_executable(label_propagation_kokkos
    main_kokkos.cpp
    label_propagation_kokkos.cpp
)

target_link_libraries(label_propagation_kokkos 
    PRIVATE 
    hypergraph_common
    Kokkos::kokkos
)

target_include_directories(label_propagation_kokkos PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    .
)