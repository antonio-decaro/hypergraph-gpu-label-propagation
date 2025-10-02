#include "label_propagation_sycl.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

LabelPropagationSYCL::LabelPropagationSYCL(sycl::queue& queue) : queue_(queue) {     
     std::cout << "SYCL device: " << queue_.get_device().get_info<sycl::info::device::name>() << "\n";
     std::cout << "SYCL platform: " << queue_.get_device().get_platform().get_info<sycl::info::platform::name>() << "\n";
}

LabelPropagationSYCL::~LabelPropagationSYCL() = default;

int LabelPropagationSYCL::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running SYCL label propagation\n";
    
    // Flatten the hypergraph for GPU processing
    auto hg = create_device_hypergraph(hypergraph);
    size_t* changes = sycl::malloc_shared<size_t>(1, queue_);
    
    int iteration = 0;

    try {
        // Create SYCL arrays for labels
        Hypergraph::Label* vertex_labels = sycl::malloc_device<Hypergraph::Label>(hypergraph.get_num_vertices(), queue_);
        Hypergraph::Label* edge_labels = sycl::malloc_device<Hypergraph::Label>(hypergraph.get_num_edges(), queue_);

        queue_.copy(hypergraph.get_labels().data(), vertex_labels, hypergraph.get_num_vertices()).wait();

        for (iteration = 0; iteration < max_iterations; ++iteration) {
            bool converged = run_iteration_sycl(hg, vertex_labels, edge_labels, changes, tolerance);

            if (converged) {
                std::cout << "Converged after " << iteration + 1 << " iterations\n";
                break;
            }

            if ((iteration + 1) % 10 == 0) {
                std::cout << "Iteration " << iteration + 1 << " completed\n";
            }
        }

        // Copy results back
        std::vector<Hypergraph::Label> final_labels(hypergraph.get_num_vertices());
        queue_.copy(vertex_labels, final_labels.data(), hypergraph.get_num_vertices()).wait();
        hypergraph.set_labels(final_labels);

    } catch (...) {
        cleanup_flat_hypergraph(hg);
        sycl::free(changes, queue_);
        throw;
    }

    cleanup_flat_hypergraph(hg);
    sycl::free(changes, queue_);
    return iteration + 1;
}

bool LabelPropagationSYCL::run_iteration_sycl(const DeviceFlatHypergraph& flat_hg,
                                              Hypergraph::Label* vertex_labels,
                                              Hypergraph::Label* edge_labels,
                                              std::size_t* changes,
                                              double tolerance) {
    *changes = 0;

    constexpr int MAX_LABELS = 10;  // Adjust based on expected label range

    // Submit label propagation kernel
    try {
        // First update edge labels based on vertex labels
        auto e = queue_.submit([&](sycl::handler& h) {
            Hypergraph::VertexId* edge_vertices = flat_hg.edge_vertices;
            std::size_t* edge_offsets = flat_hg.edge_offsets;
            Hypergraph::EdgeId* vertex_edges = flat_hg.vertex_edges;
            std::size_t* vertex_offsets = flat_hg.vertex_offsets;
            std::size_t* edge_sizes = flat_hg.edge_sizes;
            std::size_t* changes_ptr = changes;

            h.parallel_for(sycl::range<1>(flat_hg.num_edges), [=](sycl::id<1> idx) {
                std::size_t e = idx[0];

                // Get incident vertices for edge_id
                const std::size_t vertex_start = edge_offsets[e];
                const std::size_t vertex_end = edge_offsets[e + 1];
                const std::size_t edge_size = vertex_end - vertex_start;

                // Count label frequencies with weights
                float label_weights[MAX_LABELS] = {0.0f};

                // Process each incident vertex
                for (std::size_t vertex_id = vertex_start; vertex_id < vertex_end; ++vertex_id) {
                    auto vertex = edge_vertices[vertex_id];
                    std::size_t vertices_start = edge_offsets[e];

                    auto label = vertex_labels[vertex];
                    label_weights[label] += 1.0f;
                }

                // Find label with maximum weight
                Hypergraph::Label best_label = edge_labels[e];
                float max_weight = 0.0f;

                #pragma unroll
                for (int label = 0; label < MAX_LABELS; ++label) {
                    if (label_weights[label] > max_weight) {
                        max_weight = label_weights[label];
                        best_label = label;
                    }
                }

                edge_labels[e] = best_label;
            });
        });
        
        // Then update vertex labels based on edge labels
        queue_.submit([&](sycl::handler& h) {
            h.depends_on(e);
            Hypergraph::VertexId* edge_vertices = flat_hg.edge_vertices;
            std::size_t* edge_offsets = flat_hg.edge_offsets;
            Hypergraph::EdgeId* vertex_edges = flat_hg.vertex_edges;
            std::size_t* vertex_offsets = flat_hg.vertex_offsets;
            std::size_t* edge_sizes = flat_hg.edge_sizes;
            std::size_t* changes_ptr = changes;

            h.parallel_for(sycl::range<1>(flat_hg.num_vertices), [=](sycl::id<1> idx) {
                std::size_t v = idx[0];

                // Get incident vertices for edge_id
                const std::size_t edge_start = vertex_offsets[v];
                const std::size_t edge_end = vertex_offsets[v + 1];

                // Count label frequencies with weights
                float label_weights[MAX_LABELS] = {0.0f};

                // Process each incident vertex
                for (std::size_t edge_id = edge_start; edge_id < edge_end; ++edge_id) {
                    auto edge = vertex_edges[edge_id];

                    auto label = edge_labels[edge];
                    label_weights[label] += 1.0f;
                }
                

                // Find label with maximum weight
                Hypergraph::Label best_label = edge_labels[v];
                float max_weight = 0.0f;

                #pragma unroll
                for (int label = 0; label < MAX_LABELS; ++label) {
                    if (label_weights[label] > max_weight) {
                        max_weight = label_weights[label];
                        best_label = label;
                    }
                }

                // Count changes for convergence check
                if (vertex_labels[v] != best_label) {
                    vertex_labels[v] = best_label;
                    sycl::atomic_ref<std::size_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                        atomic_changes(*changes_ptr);
                    atomic_changes.fetch_add(1);
                }
            });
        }).wait();
        
    } catch (...) {
        throw;
    }

    // Check convergence
    std::size_t changes_count = *changes;
    double change_ratio = static_cast<double>(changes_count) / static_cast<double>(flat_hg.num_vertices);
    return change_ratio < tolerance;
}
