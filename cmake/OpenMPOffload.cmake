include_guard(GLOBAL)

# Options for OpenMP target offload
if(NOT DEFINED ENABLE_OPENMP_OFFLOAD)
  set(ENABLE_OPENMP_OFFLOAD OFF CACHE BOOL "Enable OpenMP target offloading (GPU)")
endif()
if(NOT DEFINED OMP_OFFLOAD_VENDOR)
  set(OMP_OFFLOAD_VENDOR "NVIDIA" CACHE STRING "OpenMP offload target vendor: NVIDIA, AMD, or INTEL")
  set_property(CACHE OMP_OFFLOAD_VENDOR PROPERTY STRINGS "NVIDIA" "AMD" "INTEL")
endif()
if(NOT DEFINED OMP_TARGET_ARCH)
  set(OMP_TARGET_ARCH "" CACHE STRING "GPU architecture (e.g., sm_80, gfx90a). Optional")
endif()

# Apply OpenMP offload flags and helpful compile definitions to a target
function(omp_configure_offload target_name)
  if(NOT ENABLE_OPENMP_OFFLOAD)
    return()
  endif()

  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "omp_configure_offload: target '${target_name}' does not exist")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|LLVM|AppleClang|IntelLLVM")
    set(_triple "")
    if(OMP_OFFLOAD_VENDOR STREQUAL "NVIDIA")
      set(_triple "nvptx64-nvidia-cuda")
    elseif(OMP_OFFLOAD_VENDOR STREQUAL "AMD")
      set(_triple "amdgcn-amd-amdhsa")
    elseif(OMP_OFFLOAD_VENDOR STREQUAL "INTEL")
      set(_triple "spir64")
    else()
      message(FATAL_ERROR "Unknown OMP_OFFLOAD_VENDOR='${OMP_OFFLOAD_VENDOR}'. Use NVIDIA, AMD, or INTEL.")
    endif()

    set(_flags "-fopenmp" "-fopenmp-targets=${_triple}")
    # Optional per-arch tuning for NVIDIA/AMD with Clang/IntelLLVM
    if(OMP_TARGET_ARCH AND (OMP_OFFLOAD_VENDOR STREQUAL "NVIDIA" OR OMP_OFFLOAD_VENDOR STREQUAL "AMD"))
      list(APPEND _flags "--offload-arch=${OMP_TARGET_ARCH}")
    endif()

    target_compile_options(${target_name} PRIVATE ${_flags})
    target_link_options(${target_name} PRIVATE ${_flags})

    message(STATUS "OpenMP offload enabled: vendor=${OMP_OFFLOAD_VENDOR} triple=${_triple} arch='${OMP_TARGET_ARCH}' for target ${target_name}")
  else()
    message(WARNING "ENABLE_OPENMP_OFFLOAD requested but compiler '${CMAKE_CXX_COMPILER_ID}' is not Clang/IntelLLVM. Offload flags not applied to ${target_name}.")
  endif()
endfunction()

