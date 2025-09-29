#include "label_propagation_sycl.hpp"
#include <iostream>
#include <unordered_map>
#include <algorithm>

LabelPropagationSYCL::LabelPropagationSYCL(sycl::device_selector* device_selector) {
    try {
        if (device_selector) {
            queue_ = sycl::queue(*device_selector);
        } else {
            // Try GPU first, fallback to CPU
            try {
                queue_ = sycl::queue(sycl::gpu_selector_v);
            } catch (...) {
                queue_ = sycl::queue(sycl::cpu_selector_v);
            }
        }
        device_ = queue_.get_device();
        
        std::cout << "SYCL device: " << device_.get_info<sycl::info::device::name>() << "\n";
        std::cout << "SYCL platform: " << device_.get_platform().get_info<sycl::info::platform::name>() << "\n";
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        throw;
    }
}

LabelPropagationSYCL::~LabelPropagationSYCL() = default;

int LabelPropagationSYCL::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running SYCL label propagation\n";
    
    // Flatten the hypergraph for GPU processing
    auto flat_hg = flatten_hypergraph(hypergraph);
    
    // Initialize labels
    auto labels = hypergraph.get_labels();
    
    // Create SYCL buffers
    sycl::buffer<Hypergraph::Label, 1> current_labels(labels.data(), sycl::range<1>(labels.size()));
    sycl::buffer<Hypergraph::Label, 1> new_labels(sycl::range<1>(labels.size()));

    int iteration = 0;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        bool converged = run_iteration_sycl(flat_hg, current_labels, new_labels, tolerance);
        
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
    {
        auto accessor = current_labels.get_host_access(sycl::read_only);
        std::vector<Hypergraph::Label> final_labels(accessor.get_pointer(), 
                                                    accessor.get_pointer() + labels.size());
        hypergraph.set_labels(final_labels);
    }

    return iteration + 1;
}

LabelPropagationSYCL::FlatHypergraph LabelPropagationSYCL::flatten_hypergraph(const Hypergraph& hypergraph) {
    FlatHypergraph flat_hg;
    flat_hg.num_vertices = hypergraph.get_num_vertices();
    flat_hg.num_edges = hypergraph.get_num_edges();
    
    // Flatten hyperedges
    flat_hg.edge_offsets.push_back(0);
    for (std::size_t e = 0; e < flat_hg.num_edges; ++e) {
        const auto& vertices = hypergraph.get_hyperedge(e);
        flat_hg.edge_sizes.push_back(vertices.size());
        
        for (auto v : vertices) {
            flat_hg.edge_vertices.push_back(v);
        }
        flat_hg.edge_offsets.push_back(flat_hg.edge_vertices.size());
    }
    
    // Flatten vertex incident edges
    flat_hg.vertex_offsets.push_back(0);
    for (std::size_t v = 0; v < flat_hg.num_vertices; ++v) {
        const auto& edges = hypergraph.get_incident_edges(v);
        
        for (auto e : edges) {
            flat_hg.vertex_edges.push_back(e);
        }
        flat_hg.vertex_offsets.push_back(flat_hg.vertex_edges.size());
    }
    
    return flat_hg;
}

bool LabelPropagationSYCL::run_iteration_sycl(const FlatHypergraph& flat_hg,
                                              sycl::buffer<Hypergraph::Label, 1>& current_labels,
                                              sycl::buffer<Hypergraph::Label, 1>& new_labels,
                                              double tolerance) {
    
    // Create buffers for flattened data
    sycl::buffer<Hypergraph::VertexId, 1> edge_vertices_buf(flat_hg.edge_vertices.data(), 
                                                            sycl::range<1>(flat_hg.edge_vertices.size()));
    sycl::buffer<std::size_t, 1> edge_offsets_buf(flat_hg.edge_offsets.data(), 
                                                   sycl::range<1>(flat_hg.edge_offsets.size()));
    sycl::buffer<Hypergraph::EdgeId, 1> vertex_edges_buf(flat_hg.vertex_edges.data(), 
                                                         sycl::range<1>(flat_hg.vertex_edges.size()));
    sycl::buffer<std::size_t, 1> vertex_offsets_buf(flat_hg.vertex_offsets.data(), 
                                                     sycl::range<1>(flat_hg.vertex_offsets.size()));
    sycl::buffer<std::size_t, 1> edge_sizes_buf(flat_hg.edge_sizes.data(), 
                                                sycl::range<1>(flat_hg.edge_sizes.size()));
    
    // Buffer for counting changes
    sycl::buffer<std::size_t, 1> changes_buf(sycl::range<1>(1));
    {
        auto accessor = changes_buf.get_host_access();
        accessor[0] = 0;
    }

    // Submit label propagation kernel
    queue_.submit([&](sycl::handler& h) {
        auto current_acc = current_labels.get_access<sycl::access::mode::read>(h);
        auto new_acc = new_labels.get_access<sycl::access::mode::write>(h);
        auto edge_vertices_acc = edge_vertices_buf.get_access<sycl::access::mode::read>(h);
        auto edge_offsets_acc = edge_offsets_buf.get_access<sycl::access::mode::read>(h);
        auto vertex_edges_acc = vertex_edges_buf.get_access<sycl::access::mode::read>(h);
        auto vertex_offsets_acc = vertex_offsets_buf.get_access<sycl::access::mode::read>(h);
        auto edge_sizes_acc = edge_sizes_buf.get_access<sycl::access::mode::read>(h);
        auto changes_acc = changes_buf.get_access<sycl::access::mode::atomic>(h);

        h.parallel_for(sycl::range<1>(flat_hg.num_vertices), [=](sycl::id<1> idx) {
            std::size_t v = idx[0];
            
            // Get incident edges for vertex v
            std::size_t edge_start = vertex_offsets_acc[v];
            std::size_t edge_end = vertex_offsets_acc[v + 1];
            
            if (edge_start == edge_end) {
                new_acc[v] = current_acc[v];  // Keep current label if isolated
                return;
            }

            // Count label frequencies with weights
            constexpr int MAX_LABELS = 1000;  // Adjust based on expected label range
            float label_weights[MAX_LABELS] = {0.0f};
            
            // Process each incident edge
            for (std::size_t edge_idx = edge_start; edge_idx < edge_end; ++edge_idx) {
                auto edge_id = vertex_edges_acc[edge_idx];
                std::size_t vertices_start = edge_offsets_acc[edge_id];
                std::size_t vertices_end = edge_offsets_acc[edge_id + 1];
                std::size_t edge_size = vertices_end - vertices_start;
                
                float weight = 1.0f / static_cast<float>(edge_size);
                
                // Add weights for all neighbors in this edge
                for (std::size_t vert_idx = vertices_start; vert_idx < vertices_end; ++vert_idx) {
                    auto neighbor = edge_vertices_acc[vert_idx];
                    if (neighbor != v) {
                        auto label = current_acc[neighbor];
                        if (label >= 0 && label < MAX_LABELS) {
                            label_weights[label] += weight;
                        }
                    }
                }
            }
            
            // Find label with maximum weight
            Hypergraph::Label best_label = current_acc[v];
            float max_weight = 0.0f;
            
            for (int label = 0; label < MAX_LABELS; ++label) {
                if (label_weights[label] > max_weight) {
                    max_weight = label_weights[label];
                    best_label = label;
                }
            }
            
            new_acc[v] = best_label;
            
            // Count changes for convergence check
            if (current_acc[v] != best_label) {
                sycl::atomic_ref<std::size_t, sycl::memory_order::relaxed, 
                                sycl::memory_scope::device> atomic_changes(changes_acc[0]);
                atomic_changes++;
            }
        });
    }).wait();

    // Check convergence
    std::size_t changes;
    {
        auto accessor = changes_buf.get_host_access();
        changes = accessor[0];
    }
    
    double change_ratio = static_cast<double>(changes) / static_cast<double>(flat_hg.num_vertices);
    return change_ratio < tolerance;
}