#include "pagerank_cuda.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

constexpr int MAX_CUDA_BLOCK_SIZE = 1024;

// Edge phase: h[e] = sum_{u in e} p[u] / d(u)
__global__ void pagerank_edge_kernel(
    const Hypergraph::VertexId* edge_vertices,
    const std::size_t* edge_offsets,
    const std::size_t* vertex_offsets,
    const float* p,
    float* h,
    std::size_t num_edges)
{
    const std::size_t e = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    float acc = 0.0f;
    const std::size_t v_begin = edge_offsets[e];
    const std::size_t v_end   = edge_offsets[e + 1];
    for (std::size_t i = v_begin; i < v_end; ++i) {
        const std::size_t u = edge_vertices[i];
        const float du = static_cast<float>(vertex_offsets[u + 1] - vertex_offsets[u]);
        if (du > 0.0f) acc += p[u] / du;
    }
    h[e] = acc;
}

// Vertex phase: p[v] = (1-alpha)/|V| + alpha * sum_{e in v} (1/delta(e)) * h[e]
__global__ void pagerank_vertex_kernel(
    const Hypergraph::EdgeId* vertex_edges,
    const std::size_t* vertex_offsets,
    const std::size_t* edge_sizes,
    const float* h,
    float* p,
    float* diff,
    float alpha,
    float inv_n,
    std::size_t num_vertices)
{
    const std::size_t v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    float acc = 0.0f;
    const std::size_t e_begin = vertex_offsets[v];
    const std::size_t e_end   = vertex_offsets[v + 1];
    for (std::size_t i = e_begin; i < e_end; ++i) {
        const std::size_t e = vertex_edges[i];
        const float delta_e = static_cast<float>(edge_sizes[e]);
        if (delta_e > 0.0f) acc += h[e] / delta_e;
    }

    const float p_new = (1.0f - alpha) * inv_n + alpha * acc;
    atomicAdd(diff, fabsf(p_new - p[v]));
    p[v] = p_new;
}

} // namespace

PageRankCUDA::PageRankCUDA(const CLI::DeviceOptions& device, double alpha) : PageRankAlgorithm(device, alpha) {
    check_cuda(cudaGetDevice(&device_id_), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device_id_), "cudaGetDeviceProperties");
    max_threads_per_block_ = std::min(prop.maxThreadsPerBlock, MAX_CUDA_BLOCK_SIZE);

    void* warmup = nullptr;
    check_cuda(cudaMalloc(&warmup, 1), "cudaMalloc(warmup)");
    check_cuda(cudaFree(warmup), "cudaFree(warmup)");

    std::cout << "CUDA device: " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
}

PageRankCUDA::~PageRankCUDA() = default;

void PageRankCUDA::check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + context + ": " + cudaGetErrorString(err));
    }
}

PageRankCUDA::DeviceFlatHypergraph PageRankCUDA::create_device_hypergraph(const Hypergraph& hypergraph) {
    DeviceFlatHypergraph device_hg;
    auto flat_hg = hypergraph.flatten();

    const auto copy_vector = [&](auto& host_vec, auto*& device_ptr) {
        using ValueType = typename std::remove_reference<decltype(host_vec)>::type::value_type;
        if (host_vec.empty()) { device_ptr = nullptr; return; }
        ValueType* raw_ptr = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&raw_ptr), host_vec.size() * sizeof(ValueType)), "cudaMalloc");
        check_cuda(cudaMemcpy(raw_ptr, host_vec.data(), host_vec.size() * sizeof(ValueType), cudaMemcpyHostToDevice), "cudaMemcpy");
        device_ptr = raw_ptr;
    };

    copy_vector(flat_hg.edge_vertices,  device_hg.edge_vertices);
    copy_vector(flat_hg.edge_offsets,   device_hg.edge_offsets);
    copy_vector(flat_hg.vertex_edges,   device_hg.vertex_edges);
    copy_vector(flat_hg.vertex_offsets, device_hg.vertex_offsets);
    copy_vector(flat_hg.edge_sizes,     device_hg.edge_sizes);
    device_hg.num_vertices = flat_hg.num_vertices;
    device_hg.num_edges    = flat_hg.num_edges;
    return device_hg;
}

void PageRankCUDA::destroy_device_hypergraph(DeviceFlatHypergraph& flat_hg) {
    auto safe_free = [](auto*& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } };
    safe_free(flat_hg.edge_vertices);
    safe_free(flat_hg.edge_offsets);
    safe_free(flat_hg.vertex_edges);
    safe_free(flat_hg.vertex_offsets);
    safe_free(flat_hg.edge_sizes);
}

PerformanceMeasurer PageRankCUDA::run(const Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running CUDA PageRank (alpha=" << alpha_ << ")\n";

    PerformanceMeasurer perf;
    const auto overall_start = PerformanceMeasurer::clock::now();

    const std::size_t num_vertices = hypergraph.get_num_vertices();
    const std::size_t num_edges    = hypergraph.get_num_edges();

    if (num_vertices == 0 || num_edges == 0) {
        std::cout << "Empty hypergraph detected; nothing to compute.\n";
        perf.set_iterations(0);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
        return perf;
    }

    DeviceFlatHypergraph flat_hg{};
    float* d_p    = nullptr;
    float* d_h    = nullptr;
    float* d_diff = nullptr;

    auto cleanup = [&]() {
        auto safe_free = [](float*& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } };
        safe_free(d_p);
        safe_free(d_h);
        safe_free(d_diff);
        destroy_device_hypergraph(flat_hg);
    };

    try {
        const auto setup_start = PerformanceMeasurer::clock::now();

        flat_hg = create_device_hypergraph(hypergraph);

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_p),    num_vertices * sizeof(float)), "cudaMalloc(p)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_h),    num_edges    * sizeof(float)), "cudaMalloc(h)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_diff),                sizeof(float)), "cudaMalloc(diff)");

        const float inv_n = 1.0f / static_cast<float>(num_vertices);
        std::vector<float> h_p(num_vertices, inv_n);
        check_cuda(cudaMemcpy(d_p, h_p.data(), num_vertices * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(p_init)");

        const auto setup_end = PerformanceMeasurer::clock::now();
        perf.add_moment("setup", setup_end - setup_start);

        int iterations_completed = 0;
        bool converged = false;

        // First iteration timed separately: isolates first-kernel launch overhead
        const auto init_start = PerformanceMeasurer::clock::now();
        if (max_iterations > 0) {
            converged = run_iteration(flat_hg, d_p, d_h, d_diff, tolerance);
            if (converged) {
                std::cout << "Converged after 1 iterations\n";
                iterations_completed = 1;
            }
        }
        const auto init_end = PerformanceMeasurer::clock::now();
        perf.add_moment("init", init_end - init_start);

        const auto iteration_start = PerformanceMeasurer::clock::now();
        for (int iteration = 1; !converged && iteration < max_iterations; ++iteration) {
            converged = run_iteration(flat_hg, d_p, d_h, d_diff, tolerance);
            if (converged) {
                std::cout << "Converged after " << iteration + 1 << " iterations\n";
                iterations_completed = iteration + 1;
            }
            if (!converged && (iteration + 1) % 10 == 0) { std::cout << "Iteration " << iteration + 1 << " completed\n"; }
        }
        if (!converged) { iterations_completed = max_iterations; }

        const auto iteration_end = PerformanceMeasurer::clock::now();
        perf.add_moment("iterations", iteration_end - iteration_start);

        const auto finalize_start = PerformanceMeasurer::clock::now();
        scores_.resize(num_vertices);
        check_cuda(cudaMemcpy(scores_.data(), d_p, num_vertices * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(scores)");
        const auto finalize_end = PerformanceMeasurer::clock::now();
        perf.add_moment("finalize", finalize_end - finalize_start);

        perf.set_iterations(iterations_completed);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);

        cleanup();
        return perf;
    } catch (...) {
        cleanup();
        throw;
    }
}

bool PageRankCUDA::run_iteration(
    const DeviceFlatHypergraph& flat_hg, float* d_p, float* d_h, float* d_diff, double tolerance)
{
    int threads = device_.workgroup_size > 0 ? static_cast<int>(device_.workgroup_size) : 256;
    threads = std::max(1, std::min(threads, max_threads_per_block_));
    int pow2 = 1;
    while (pow2 * 2 <= threads) pow2 *= 2;
    threads = pow2;

    const float alpha_f = static_cast<float>(alpha_);
    const float inv_n   = 1.0f / static_cast<float>(flat_hg.num_vertices);
    const dim3 block(static_cast<unsigned int>(threads));

    // Edge phase
    const dim3 edge_grid((static_cast<unsigned int>(flat_hg.num_edges) + block.x - 1) / block.x);
    pagerank_edge_kernel<<<edge_grid, block>>>(
        flat_hg.edge_vertices, flat_hg.edge_offsets,
        flat_hg.vertex_offsets,
        d_p, d_h, flat_hg.num_edges);
    check_cuda(cudaGetLastError(), "pagerank_edge_kernel");

    check_cuda(cudaMemset(d_diff, 0, sizeof(float)), "cudaMemset(diff)");

    // Vertex phase
    const dim3 vertex_grid((static_cast<unsigned int>(flat_hg.num_vertices) + block.x - 1) / block.x);
    pagerank_vertex_kernel<<<vertex_grid, block>>>(
        flat_hg.vertex_edges, flat_hg.vertex_offsets,
        flat_hg.edge_sizes,
        d_h, d_p, d_diff,
        alpha_f, inv_n, flat_hg.num_vertices);
    check_cuda(cudaGetLastError(), "pagerank_vertex_kernel");

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    float diff_val = 0.0f;
    check_cuda(cudaMemcpy(&diff_val, d_diff, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(diff)");
    return static_cast<double>(diff_val) < tolerance;
}
