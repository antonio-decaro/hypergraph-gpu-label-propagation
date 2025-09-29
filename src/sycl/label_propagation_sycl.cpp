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
        Hypergraph::Label* current_labels = sycl::malloc_shared<Hypergraph::Label>(hypergraph.get_labels().size(), queue_);
        Hypergraph::Label* new_labels = sycl::malloc_shared<Hypergraph::Label>(hypergraph.get_labels().size(), queue_);

        queue_.copy(hypergraph.get_labels().data(), current_labels, hypergraph.get_labels().size()).wait();

        for (iteration = 0; iteration < max_iterations; ++iteration) {
            bool converged = run_iteration_sycl(hg, current_labels, new_labels, changes, tolerance);

            if (converged) {
                std::cout << "Converged after " << iteration + 1 << " iterations\n";
                break;
            }

            // Swap buffers
            std::swap(current_labels, new_labels);

            if ((iteration + 1) % 10 == 0) {
                std::cout << "Iteration " << iteration + 1 << " completed\n";
            }
        }

        // Copy results back
        std::vector<Hypergraph::Label> final_labels(hypergraph.get_num_vertices());
        queue_.copy(new_labels, final_labels.data(), hypergraph.get_num_vertices()).wait();
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
                                              const Hypergraph::Label* current_labels,
                                              Hypergraph::Label* new_labels,
                                              std::size_t* changes,
                                              double tolerance) {
    *changes = 0;

    // Submit label propagation kernel
    try {
        queue_.submit([&](sycl::handler& h) {
            Hypergraph::VertexId* edge_vertices = flat_hg.edge_vertices;
            std::size_t* edge_offsets = flat_hg.edge_offsets;
            Hypergraph::EdgeId* vertex_edges = flat_hg.vertex_edges;
            std::size_t* vertex_offsets = flat_hg.vertex_offsets;
            std::size_t* edge_sizes = flat_hg.edge_sizes;
            std::size_t* changes_ptr = changes;

            h.parallel_for(sycl::range<1>(flat_hg.num_vertices), [=](sycl::id<1> idx) {
                std::size_t v = idx[0];

                // Get incident edges for vertex v
                std::size_t edge_start = vertex_offsets[v];
                std::size_t edge_end = vertex_offsets[v + 1];

                if (edge_start == edge_end) {
                    new_labels[v] = current_labels[v];  // Keep current label if isolated
                    return;
                }

                // Count label frequencies with weights
                constexpr int MAX_LABELS = 1000;  // Adjust based on expected label range
                float label_weights[MAX_LABELS] = {0.0f};

                // Process each incident edge
                for (std::size_t edge_idx = edge_start; edge_idx < edge_end; ++edge_idx) {
                    auto edge_id = vertex_edges[edge_idx];
                    std::size_t vertices_start = edge_offsets[edge_id];
                    std::size_t edge_size = edge_sizes ? edge_sizes[edge_id]
                                                       : edge_offsets[edge_id + 1] - vertices_start;

                    if (edge_size == 0) {
                        continue;
                    }

                    float weight = 1.0f / static_cast<float>(edge_size);
                    std::size_t vertices_end = vertices_start + edge_size;

                    // Add weights for all neighbors in this edge
                    for (std::size_t vert_idx = vertices_start; vert_idx < vertices_end; ++vert_idx) {
                        auto neighbor = edge_vertices[vert_idx];
                        if (neighbor != v) {
                            auto label = current_labels[neighbor];
                            if (label >= 0 && label < MAX_LABELS) {
                                label_weights[label] += weight;
                            }
                        }
                    }
                }

                // Find label with maximum weight
                Hypergraph::Label best_label = current_labels[v];
                float max_weight = 0.0f;

                for (int label = 0; label < MAX_LABELS; ++label) {
                    if (label_weights[label] > max_weight) {
                        max_weight = label_weights[label];
                        best_label = label;
                    }
                }

                new_labels[v] = best_label;

                // Count changes for convergence check
                if (current_labels[v] != best_label) {
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
        sycl::free(changes, queue_);
        throw;
    }

    // Check convergence
    std::size_t changes_count = *changes;
    double change_ratio = static_cast<double>(changes_count) / static_cast<double>(flat_hg.num_vertices);
    return change_ratio < tolerance;
}
