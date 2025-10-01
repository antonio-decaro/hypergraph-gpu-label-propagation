include_guard(GLOBAL)

# Apply OpenMP offload flags and helpful compile definitions to a target
function(omp_configure_offload target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "omp_configure_offload: target '${target_name}' does not exist")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|LLVM|AppleClang|IntelLLVM")
    set(_triple "")
    if(OFFLOAD_VENDOR STREQUAL "NVIDIA")
      set(_triple "nvptx64-nvidia-cuda")
    elseif(OFFLOAD_VENDOR STREQUAL "AMD")
      set(_triple "amdgcn-amd-amdhsa")
    elseif(OFFLOAD_VENDOR STREQUAL "INTEL")
      set(_triple "spir64")
    else()
      message(FATAL_ERROR "Unknown OFFLOAD_VENDOR='${OFFLOAD_VENDOR}'. Use NVIDIA, AMD, or INTEL.")
    endif()

    set(_flags "-fopenmp" "-fopenmp-targets=${_triple}")
    # Optional per-arch tuning for NVIDIA/AMD with Clang/IntelLLVM
    if(OFFLOAD_TARGET AND NOT OFFLOAD_TARGET STREQUAL "none") 
      list(APPEND _flags "--offload-arch=${OFFLOAD_TARGET}")
    endif()

    target_compile_options(${target_name} PRIVATE ${_flags})
    target_link_options(${target_name} PRIVATE ${_flags})

    message(STATUS "OpenMP offload enabled: vendor=${OFFLOAD_VENDOR} triple=${_triple} arch='${OFFLOAD_TARGET}' for target ${target_name}")
  else()
    message(WARNING "ENABLE_OPENMP_OFFLOAD requested but compiler '${CMAKE_CXX_COMPILER_ID}' is not Clang/IntelLLVM. Offload flags not applied to ${target_name}.")
  endif()
endfunction()

