include_guard(GLOBAL)

# Apply SYCL offload flags and helpful compile definitions to a target
function(sycl_configure_offload target_name)
    if (OFFLOAD_TARGET STREQUAL "none")
        message(WARNING "OFFLOAD_TARGET is set to 'none'. Please set it to a valid SYCL target (e.g., spir64, nvptx64-nvidia-cuda, etc.) for offloading.")
        if (OFFLOAD_VENDOR STREQUAL "NVIDIA")
            set(SYCL_TARGET "nvptx64-nvidia-cuda")
        elseif (OFFLOAD_VENDOR STREQUAL "AMD")
            message(ERROR "Please set OFFLOAD_TARGET to a valid AMD SYCL target (e.g., gfx908, gfx90a, etc.)")
        elseif (OFFLOAD_VENDOR STREQUAL "INTEL")
            set(SYCL_TARGET "spir64")
        endif()
    else()
        if (OFFLOAD_VENDOR STREQUAL "NVIDIA")
            set(SYCL_TARGET "nvidia_gpu_${OFFLOAD_TARGET}")
        elseif (OFFLOAD_VENDOR STREQUAL "AMD")
            set(SYCL_TARGET "amd_gpu_${OFFLOAD_TARGET}")
        endif()
    endif()

    target_compile_options(label_propagation_sycl PRIVATE 
        -fsycl
        -fsycl-targets=${SYCL_TARGET}
    )

    target_link_options(label_propagation_sycl PRIVATE 
        -fsycl
        -fsycl-targets=${SYCL_TARGET}
    )
endfunction()
