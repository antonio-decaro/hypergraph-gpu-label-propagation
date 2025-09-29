#include "label_propagation_kokkos.hpp"
#include <iostream>
#include <algorithm>

LabelPropagationKokkos::LabelPropagationKokkos() : kokkos_initialized_(false) {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        kokkos_initialized_ = true;
        std::cout << "Kokkos initialized\n";
    }
    
    std::cout << "Kokkos execution space: " << typeid(ExecutionSpace).name() << "\n";
    std::cout << "Kokkos memory space: " << typeid(MemorySpace).name() << "\n";
}

LabelPropagationKokkos::~LabelPropagationKokkos() {
    if (kokkos_initialized_ && Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
}

int LabelPropagationKokkos::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running Kokkos label propagation\n";
    
    // Convert to Kokkos representation
    auto kokkos_hg = create_kokkos_hypergraph(hypergraph);
    
    // Initialize labels
    auto labels = hypergraph.get_labels();
    LabelView current_labels("current_labels", labels.size());
    LabelView new_labels("new_labels", labels.size());
    
    // Copy initial labels to device
    auto host_labels = Kokkos::create_mirror_view(current_labels);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        host_labels(i) = labels[i];
    }
    Kokkos::deep_copy(current_labels, host_labels);

    int iteration = 0;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        bool converged = run_iteration_kokkos(kokkos_hg, current_labels, new_labels, tolerance);
        
        if (converged) {
            std::cout << "Converged after " << iteration + 1 << " iterations\n";
            break;
        }

        // Swap views
        std::swap(current_labels, new_labels);
        
        if ((iteration + 1) % 10 == 0) {
            std::cout << "Iteration " << iteration + 1 << " completed\n";
        }
    }

    // Copy results back to host
    Kokkos::deep_copy(host_labels, current_labels);
    std::vector<Hypergraph::Label> final_labels(labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        final_labels[i] = host_labels(i);
    }
    hypergraph.set_labels(final_labels);

    return iteration + 1;
}

LabelPropagationKokkos::KokkosHypergraph 
LabelPropagationKokkos::create_kokkos_hypergraph(const Hypergraph& hypergraph) {
    KokkosHypergraph kokkos_hg;
    kokkos_hg.num_vertices = hypergraph.get_num_vertices();
    kokkos_hg.num_edges = hypergraph.get_num_edges();
    
    // Calculate sizes for flattened arrays
    std::size_t total_edge_vertices = 0;
    std::size_t total_vertex_edges = 0;
    
    for (std::size_t e = 0; e < kokkos_hg.num_edges; ++e) {
        total_edge_vertices += hypergraph.get_hyperedge(e).size();
    }
    
    for (std::size_t v = 0; v < kokkos_hg.num_vertices; ++v) {
        total_vertex_edges += hypergraph.get_incident_edges(v).size();
    }
    
    // Allocate Kokkos views
    kokkos_hg.edge_vertices = VertexView("edge_vertices", total_edge_vertices);
    kokkos_hg.edge_offsets = SizeView("edge_offsets", kokkos_hg.num_edges + 1);
    kokkos_hg.vertex_edges = EdgeView("vertex_edges", total_vertex_edges);
    kokkos_hg.vertex_offsets = SizeView("vertex_offsets", kokkos_hg.num_vertices + 1);
    kokkos_hg.edge_sizes = SizeView("edge_sizes", kokkos_hg.num_edges);
    
    // Create host mirrors and fill data
    auto h_edge_vertices = Kokkos::create_mirror_view(kokkos_hg.edge_vertices);
    auto h_edge_offsets = Kokkos::create_mirror_view(kokkos_hg.edge_offsets);
    auto h_vertex_edges = Kokkos::create_mirror_view(kokkos_hg.vertex_edges);
    auto h_vertex_offsets = Kokkos::create_mirror_view(kokkos_hg.vertex_offsets);
    auto h_edge_sizes = Kokkos::create_mirror_view(kokkos_hg.edge_sizes);
    
    // Fill edge data
    std::size_t edge_vertex_idx = 0;
    h_edge_offsets(0) = 0;
    
    for (std::size_t e = 0; e < kokkos_hg.num_edges; ++e) {
        const auto& vertices = hypergraph.get_hyperedge(e);
        h_edge_sizes(e) = vertices.size();
        
        for (auto v : vertices) {
            h_edge_vertices(edge_vertex_idx++) = v;
        }
        h_edge_offsets(e + 1) = edge_vertex_idx;
    }
    
    // Fill vertex data
    std::size_t vertex_edge_idx = 0;
    h_vertex_offsets(0) = 0;
    
    for (std::size_t v = 0; v < kokkos_hg.num_vertices; ++v) {
        const auto& edges = hypergraph.get_incident_edges(v);
        
        for (auto e : edges) {
            h_vertex_edges(vertex_edge_idx++) = e;
        }
        h_vertex_offsets(v + 1) = vertex_edge_idx;
    }
    
    // Copy to device
    Kokkos::deep_copy(kokkos_hg.edge_vertices, h_edge_vertices);
    Kokkos::deep_copy(kokkos_hg.edge_offsets, h_edge_offsets);
    Kokkos::deep_copy(kokkos_hg.vertex_edges, h_vertex_edges);
    Kokkos::deep_copy(kokkos_hg.vertex_offsets, h_vertex_offsets);
    Kokkos::deep_copy(kokkos_hg.edge_sizes, h_edge_sizes);
    
    return kokkos_hg;
}

bool LabelPropagationKokkos::run_iteration_kokkos(const KokkosHypergraph& kokkos_hg,
                                                  const LabelView& current_labels,
                                                  const LabelView& new_labels,
                                                  double tolerance) {
    
    // Counter for convergence check
    CounterView changes_counter("changes_counter", 1);
    Kokkos::deep_copy(changes_counter, 0);

    // Label propagation kernel
    Kokkos::parallel_for("label_propagation", kokkos_hg.num_vertices, 
                        KOKKOS_LAMBDA(const std::size_t v) {
        
        std::size_t edge_start = kokkos_hg.vertex_offsets(v);
        std::size_t edge_end = kokkos_hg.vertex_offsets(v + 1);
        
        if (edge_start == edge_end) {
            new_labels(v) = current_labels(v);  // Keep current label if isolated
            return;
        }

        // Count label frequencies with weights
        constexpr int MAX_LABELS = 1000;  // Adjust based on expected label range
        double label_weights[MAX_LABELS];
        
        // Initialize weights
        for (int i = 0; i < MAX_LABELS; ++i) {
            label_weights[i] = 0.0;
        }
        
        // Process each incident edge
        for (std::size_t edge_idx = edge_start; edge_idx < edge_end; ++edge_idx) {
            auto edge_id = kokkos_hg.vertex_edges(edge_idx);
            std::size_t vertices_start = kokkos_hg.edge_offsets(edge_id);
            std::size_t vertices_end = kokkos_hg.edge_offsets(edge_id + 1);
            std::size_t edge_size = vertices_end - vertices_start;
            
            double weight = 1.0 / static_cast<double>(edge_size);
            
            // Add weights for all neighbors in this edge
            for (std::size_t vert_idx = vertices_start; vert_idx < vertices_end; ++vert_idx) {
                auto neighbor = kokkos_hg.edge_vertices(vert_idx);
                if (neighbor != v) {
                    auto label = current_labels(neighbor);
                    if (label >= 0 && label < MAX_LABELS) {
                        label_weights[label] += weight;
                    }
                }
            }
        }
        
        // Find label with maximum weight
        Hypergraph::Label best_label = current_labels(v);
        double max_weight = 0.0;
        
        for (int label = 0; label < MAX_LABELS; ++label) {
            if (label_weights[label] > max_weight) {
                max_weight = label_weights[label];
                best_label = label;
            }
        }
        
        new_labels(v) = best_label;
        
        // Count changes for convergence check
        if (current_labels(v) != best_label) {
            Kokkos::atomic_increment(&changes_counter(0));
        }
    });

    // Get change count
    auto host_changes = Kokkos::create_mirror_view(changes_counter);
    Kokkos::deep_copy(host_changes, changes_counter);
    std::size_t changes = host_changes(0);
    
    double change_ratio = static_cast<double>(changes) / static_cast<double>(kokkos_hg.num_vertices);
    return change_ratio < tolerance;
}