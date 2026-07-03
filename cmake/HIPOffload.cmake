include_guard(GLOBAL)

# Apply HIP offload settings and helpful compile options to a target
function(hip_configure_offload target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "hip_configure_offload: target '${target_name}' does not exist")
    endif()

    if(NOT OFFLOAD_VENDOR STREQUAL "AMD")
        message(WARNING "HIP implementation currently supports AMD GPUs; ignoring OFFLOAD_VENDOR='${OFFLOAD_VENDOR}'")
    endif()

    if(OFFLOAD_TARGET AND NOT OFFLOAD_TARGET STREQUAL "none")
        # e.g. gfx908, gfx90a, gfx942; drives both compile and device link
        set_target_properties(${target_name} PROPERTIES HIP_ARCHITECTURES "${OFFLOAD_TARGET}")
    endif()

    # Match icpx's fast-math default so HIP device code uses the same FP
    # semantics as the SYCL/CUDA builds (approximate division/sqrt)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)

    set_target_properties(${target_name} PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )
endfunction()
