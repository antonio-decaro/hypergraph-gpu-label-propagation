include_guard(GLOBAL)

function(cuda_configure_offload target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "cuda_configure_offload: target '${target_name}' does not exist")
    endif()

    if(NOT OFFLOAD_VENDOR STREQUAL "NVIDIA")
        message(WARNING "CUDA implementation currently supports NVIDIA GPUs; ignoring OFFLOAD_VENDOR='${OFFLOAD_VENDOR}'")
    endif()

    if(OFFLOAD_TARGET AND NOT OFFLOAD_TARGET STREQUAL "none")
        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${OFFLOAD_TARGET}>)
        # Also set CUDA_ARCHITECTURES so the device-link step targets the same
        # arch (otherwise it falls back to CMake's default, e.g. sm_52, and the
        # runtime must JIT from PTX)
        string(REGEX REPLACE "^sm_" "" _cuda_arch "${OFFLOAD_TARGET}")
        set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "${_cuda_arch}")
    endif()

    # Match icpx's fast-math default so CUDA and SYCL device code use the same
    # FP semantics (approximate division/sqrt)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endfunction()
