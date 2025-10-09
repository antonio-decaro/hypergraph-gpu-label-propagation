#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
// Kernel class template
template<typename T>
class TestKernel {
  private:
    T* A;
    T* B;
    T* C;
    local_accessor<T, 1> localA;

  public:
    // Constructor
    TestKernel(T* A, T* B, T* C, local_accessor<T, 1> localA) : A(A), B(B), C(C), localA(localA) {}

    void operator()(nd_item<1> it) const {
        size_t gid = it.get_global_id(0);
        size_t lid = it.get_local_id(0);
        size_t group_size = it.get_local_range(0);
        size_t group_id = it.get_group(0);

        // Load data into local memory
        localA[lid] = A[gid] + B[gid];
        it.barrier(access::fence_space::local_space);

        // Simple parallel reduction in local memory
        for (size_t stride = group_size / 2; stride > 0; stride /= 2) {
            if (lid < stride) localA[lid] += localA[lid + stride];
            it.barrier(access::fence_space::local_space);
        }

        // First thread writes group result to global memory
        if (lid == 0) C[group_id] = localA[0];
    }
};

int main() {
    constexpr size_t N = 1024;
    constexpr size_t local_size = 64;

    queue q{gpu_selector_v};
    sycl::kernel_id kernel_id = sycl::get_kernel_id<TestKernel<float>>();

    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {kernel_id});
    auto kernel = bundle.get_kernel(kernel_id);
    const auto wg_size = kernel.template ext_oneapi_get_info<syclex::info::max_work_group_size>(q);
    const sycl::range<1> local_range{wg_size};
    auto num_wg = kernel.template ext_oneapi_get_info<sycl::info::kernel::max_num_work_groups>(q);

    // auto wg_size = kernel.get_info<size_query>(q);
    // auto num_wg = kernel.get_info<num_query>(q, wg_size);
    std::cout << "Max work group size: " << wg_size << "\n";
    std::cout << "Max number of work groups: " << num_wg << std::endl;
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";


    // --- USM allocations ---
    float* A = malloc_shared<float>(N, q);
    float* B = malloc_shared<float>(N, q);
    float* C = malloc_shared<float>(N / local_size, q);

    // Initialize input data
    for (size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = 1.0f;
    }

    q.submit([&](handler& h) {
         // Local (shared) memory allocation for each work-group
         local_accessor<float, 1> localA(range<1>(local_size), h);

         h.parallel_for(nd_range<1>(num_wg * wg_size, wg_size), TestKernel<float>(A, B, C, localA));
     }).wait();

    // Display results
    std::cout << "Partial reductions (per group):\n";
    for (size_t i = 0; i < N / local_size; ++i) std::cout << "Group " << i << ": " << C[i] << "\n";

    // Free USM memory
    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}