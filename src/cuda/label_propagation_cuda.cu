#include "label_propagation_cuda.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <type_traits>
#include <vector>

namespace {

constexpr int MAX_LABELS = 10; // must cover possible label values
constexpr int MAX_CUDA_BLOCK_SIZE = 1024;

__global__ void
update_edge_labels_kernel(const Hypergraph::VertexId* edge_vertices, const std::size_t* edge_offsets, Hypergraph::Label* edge_labels, const Hypergraph::Label* vertex_labels, std::size_t num_edges) {
    const std::size_t edge = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (edge >= num_edges) { return; }

    __shared__ float shared_counts[MAX_LABELS * MAX_CUDA_BLOCK_SIZE];
    float* counts = &shared_counts[static_cast<std::size_t>(threadIdx.x) * MAX_LABELS];

    for (int i = 0; i < MAX_LABELS; ++i) { counts[i] = 0.0f; }

    const std::size_t v_begin = edge_offsets[edge];
    const std::size_t v_end = edge_offsets[edge + 1];
    for (std::size_t idx = v_begin; idx < v_end; ++idx) {
        const auto v = edge_vertices[idx];
        const int label = static_cast<int>(vertex_labels[v]);
        if (label >= 0 && label < MAX_LABELS) { counts[label] += 1.0f; }
    }

    int best_label = edge_labels[edge];
    float best_weight = -1.0f;
    for (int label = 0; label < MAX_LABELS; ++label) {
        const float weight = counts[label];
        if (weight > best_weight) {
            best_weight = weight;
            best_label = label;
        }
    }

    edge_labels[edge] = static_cast<Hypergraph::Label>(best_label);
}

__global__ void update_vertex_labels_kernel(const Hypergraph::EdgeId* vertex_edges,
                                            const std::size_t* vertex_offsets,
                                            const Hypergraph::Label* edge_labels,
                                            Hypergraph::Label* vertex_labels,
                                            unsigned int* change_flags,
                                            std::size_t num_vertices) {
    const std::size_t vertex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (vertex >= num_vertices) { return; }

    __shared__ float shared_counts[MAX_LABELS * MAX_CUDA_BLOCK_SIZE];
    float* counts = &shared_counts[static_cast<std::size_t>(threadIdx.x) * MAX_LABELS];

    for (int i = 0; i < MAX_LABELS; ++i) { counts[i] = 0.0f; }

    const std::size_t e_begin = vertex_offsets[vertex];
    const std::size_t e_end = vertex_offsets[vertex + 1];
    for (std::size_t idx = e_begin; idx < e_end; ++idx) {
        const auto edge = vertex_edges[idx];
        const int label = static_cast<int>(edge_labels[edge]);
        if (label >= 0 && label < MAX_LABELS) { counts[label] += 1.0f; }
    }

    int best_label = vertex_labels[vertex];
    float best_weight = -1.0f;
    for (int label = 0; label < MAX_LABELS; ++label) {
        const float weight = counts[label];
        if (weight > best_weight) {
            best_weight = weight;
            best_label = label;
        }
    }

    unsigned int changed = 0;
    if (vertex_labels[vertex] != static_cast<Hypergraph::Label>(best_label)) {
        vertex_labels[vertex] = static_cast<Hypergraph::Label>(best_label);
        changed = 1;
    }

    change_flags[vertex] = changed;
}

} // namespace

LabelPropagationCUDA::LabelPropagationCUDA(const CLI::DeviceOptions& device) : LabelPropagationAlgorithm(device) {
    check_cuda(cudaGetDevice(&device_id_), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device_id_), "cudaGetDeviceProperties");
    max_threads_per_block_ = std::min(prop.maxThreadsPerBlock, MAX_CUDA_BLOCK_SIZE);

    std::cout << "CUDA device: " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
}

LabelPropagationCUDA::~LabelPropagationCUDA() = default;

void LabelPropagationCUDA::check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) { throw std::runtime_error(std::string("CUDA error in ") + context + ": " + cudaGetErrorString(err)); }
}

LabelPropagationCUDA::DeviceFlatHypergraph LabelPropagationCUDA::create_device_hypergraph(const Hypergraph& hypergraph) {
    DeviceFlatHypergraph device_hg;
    auto flat_hg = hypergraph.flatten();

    const auto copy_vector = [&](auto& host_vec, auto*& device_ptr) {
        using ValueType = typename std::remove_reference<decltype(host_vec)>::type::value_type;
        if (host_vec.empty()) {
            device_ptr = nullptr;
            return;
        }
        ValueType* raw_ptr = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&raw_ptr), host_vec.size() * sizeof(ValueType)), "cudaMalloc(flattened data)");
        check_cuda(cudaMemcpy(raw_ptr, host_vec.data(), host_vec.size() * sizeof(ValueType), cudaMemcpyHostToDevice), "cudaMemcpy(flattened data)");
        device_ptr = raw_ptr;
    };

    copy_vector(flat_hg.edge_vertices, device_hg.edge_vertices);
    copy_vector(flat_hg.edge_offsets, device_hg.edge_offsets);
    copy_vector(flat_hg.vertex_edges, device_hg.vertex_edges);
    copy_vector(flat_hg.vertex_offsets, device_hg.vertex_offsets);
    copy_vector(flat_hg.edge_sizes, device_hg.edge_sizes);

    device_hg.num_vertices = flat_hg.num_vertices;
    device_hg.num_edges = flat_hg.num_edges;

    return device_hg;
}

void LabelPropagationCUDA::destroy_device_hypergraph(DeviceFlatHypergraph& flat_hg) {
    if (flat_hg.edge_vertices != nullptr) {
        cudaFree(flat_hg.edge_vertices);
        flat_hg.edge_vertices = nullptr;
    }
    if (flat_hg.edge_offsets != nullptr) {
        cudaFree(flat_hg.edge_offsets);
        flat_hg.edge_offsets = nullptr;
    }
    if (flat_hg.vertex_edges != nullptr) {
        cudaFree(flat_hg.vertex_edges);
        flat_hg.vertex_edges = nullptr;
    }
    if (flat_hg.vertex_offsets != nullptr) {
        cudaFree(flat_hg.vertex_offsets);
        flat_hg.vertex_offsets = nullptr;
    }
    if (flat_hg.edge_sizes != nullptr) {
        cudaFree(flat_hg.edge_sizes);
        flat_hg.edge_sizes = nullptr;
    }
}

PerformanceMeasurer LabelPropagationCUDA::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running CUDA label propagation\n";

    PerformanceMeasurer perf;
    const auto overall_start = PerformanceMeasurer::clock::now();

    const std::size_t num_vertices = hypergraph.get_num_vertices();
    const std::size_t num_edges = hypergraph.get_num_edges();

    if (num_vertices == 0 || num_edges == 0) {
        std::cout << "Empty hypergraph detected; nothing to compute.\n";
        perf.set_iterations(0);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
        return perf;
    }

    DeviceFlatHypergraph flat_hg{};
    Hypergraph::Label* d_vertex_labels = nullptr;
    Hypergraph::Label* d_edge_labels = nullptr;
    unsigned int* d_change_flags = nullptr;

    auto cleanup = [&]() {
        if (d_vertex_labels) {
            cudaFree(d_vertex_labels);
            d_vertex_labels = nullptr;
        }
        if (d_edge_labels) {
            cudaFree(d_edge_labels);
            d_edge_labels = nullptr;
        }
        if (d_change_flags) {
            cudaFree(d_change_flags);
            d_change_flags = nullptr;
        }
        destroy_device_hypergraph(flat_hg);
    };

    try {
        const auto setup_start = PerformanceMeasurer::clock::now();

        flat_hg = create_device_hypergraph(hypergraph);

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_vertex_labels), num_vertices * sizeof(Hypergraph::Label)), "cudaMalloc(vertex_labels)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_edge_labels), num_edges * sizeof(Hypergraph::Label)), "cudaMalloc(edge_labels)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_change_flags), num_vertices * sizeof(unsigned int)), "cudaMalloc(change_flags)");

        const auto initial_labels = hypergraph.get_labels();
        check_cuda(cudaMemcpy(d_vertex_labels, initial_labels.data(), num_vertices * sizeof(Hypergraph::Label), cudaMemcpyHostToDevice), "cudaMemcpy(vertex_labels)");
        check_cuda(cudaMemset(d_edge_labels, 0, num_edges * sizeof(Hypergraph::Label)), "cudaMemset(edge_labels)");

        const auto setup_end = PerformanceMeasurer::clock::now();
        perf.add_moment("setup", setup_end - setup_start);

        const auto iteration_start = PerformanceMeasurer::clock::now();
        int iterations_completed = 0;
        bool converged = false;
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            const bool iteration_converged = run_iteration_cuda(flat_hg, d_vertex_labels, d_edge_labels, d_change_flags, tolerance);
            if (iteration_converged) {
                std::cout << "Converged after " << iteration + 1 << " iterations\n";
                iterations_completed = iteration + 1;
                converged = true;
                break;
            }
            if ((iteration + 1) % 10 == 0) { std::cout << "Iteration " << iteration + 1 << " completed\n"; }
        }
        if (!converged) { iterations_completed = max_iterations; }

        const auto iteration_end = PerformanceMeasurer::clock::now();
        perf.add_moment("iterations", iteration_end - iteration_start);

        const auto finalize_start = PerformanceMeasurer::clock::now();
        std::vector<Hypergraph::Label> host_labels(num_vertices);
        check_cuda(cudaMemcpy(host_labels.data(), d_vertex_labels, num_vertices * sizeof(Hypergraph::Label), cudaMemcpyDeviceToHost), "cudaMemcpy(host_labels)");
        hypergraph.set_labels(host_labels);
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

bool LabelPropagationCUDA::run_iteration_cuda(
    const DeviceFlatHypergraph& flat_hg, Hypergraph::Label* d_vertex_labels, Hypergraph::Label* d_edge_labels, unsigned int* d_change_flags, double tolerance) {
    if (flat_hg.num_vertices == 0) { return true; }

    int threads = device_.workgroup_size > 0 ? static_cast<int>(device_.workgroup_size) : 256;
    threads = std::max(1, std::min(threads, max_threads_per_block_));

    int pow2_threads = 1;
    while (pow2_threads * 2 <= threads) { pow2_threads *= 2; }
    threads = pow2_threads;

    const dim3 block_dim(static_cast<unsigned int>(threads));
    const dim3 edge_grid_dim(static_cast<unsigned int>((flat_hg.num_edges + block_dim.x - 1) / block_dim.x));

    if (flat_hg.num_edges > 0) {
        update_edge_labels_kernel<<<edge_grid_dim, block_dim>>>(flat_hg.edge_vertices, flat_hg.edge_offsets, d_edge_labels, d_vertex_labels, flat_hg.num_edges);
        check_cuda(cudaGetLastError(), "update_edge_labels_kernel");
    }

    const dim3 vertex_grid_dim(static_cast<unsigned int>((flat_hg.num_vertices + block_dim.x - 1) / block_dim.x));
    if (flat_hg.num_vertices > 0) {
        update_vertex_labels_kernel<<<vertex_grid_dim, block_dim>>>(flat_hg.vertex_edges, flat_hg.vertex_offsets, d_edge_labels, d_vertex_labels, d_change_flags, flat_hg.num_vertices);
        check_cuda(cudaGetLastError(), "update_vertex_labels_kernel");
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    thrust::device_ptr<unsigned int> changes_begin(d_change_flags);
    const unsigned int change_count = thrust::reduce(thrust::device, changes_begin, changes_begin + flat_hg.num_vertices, 0u);

    const double change_ratio = static_cast<double>(change_count) / static_cast<double>(flat_hg.num_vertices);
    return change_ratio < tolerance;
}
