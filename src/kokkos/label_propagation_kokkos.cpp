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
    
    // Initialize vertex and edge labels
    auto host_init_labels = hypergraph.get_labels();
    LabelView vertex_labels("vertex_labels", host_init_labels.size());
    LabelView edge_labels("edge_labels", kokkos_hg.num_edges);

    // Copy initial vertex labels to device
    {
        auto h_vlabels = Kokkos::create_mirror_view(vertex_labels);
        for (std::size_t i = 0; i < host_init_labels.size(); ++i) {
            h_vlabels(i) = host_init_labels[i];
        }
        Kokkos::deep_copy(vertex_labels, h_vlabels);
    }
    // Initialize edge labels to zero
    Kokkos::deep_copy(edge_labels, Hypergraph::Label(0));

    int iteration = 0;
    constexpr int MAX_LABELS = 10; // keep consistent with SYCL/OpenMP
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        // Phase 1: update edge labels from incident vertex labels (unweighted counts)
        Kokkos::parallel_for("edge_update", kokkos_hg.num_edges, KOKKOS_LAMBDA(const std::size_t e) {
            float counts[MAX_LABELS];
            for (int i = 0; i < MAX_LABELS; ++i) counts[i] = 0.0f;

            const std::size_t v_begin = kokkos_hg.edge_offsets(e);
            const std::size_t v_end   = kokkos_hg.edge_offsets(e + 1);
            for (std::size_t j = v_begin; j < v_end; ++j) {
                const auto u = kokkos_hg.edge_vertices(j);
                const int lab = static_cast<int>(vertex_labels(u));
                if (lab >= 0 && lab < MAX_LABELS) counts[lab] += 1.0f;
            }

            int best = edge_labels(e);
            float best_w = -1.0f;
            for (int lab = 0; lab < MAX_LABELS; ++lab) {
                if (counts[lab] > best_w) {
                    best_w = counts[lab];
                    best = lab;
                }
            }
            edge_labels(e) = static_cast<Hypergraph::Label>(best);
        });

        // Phase 2: update vertex labels from incident edge labels and count changes
        std::size_t changes = 0;
        Kokkos::parallel_reduce("vertex_update", kokkos_hg.num_vertices,
            KOKKOS_LAMBDA(const std::size_t v, std::size_t& local_changes) {
                float counts[MAX_LABELS];
                for (int i = 0; i < MAX_LABELS; ++i) counts[i] = 0.0f;

                const std::size_t e_begin = kokkos_hg.vertex_offsets(v);
                const std::size_t e_end   = kokkos_hg.vertex_offsets(v + 1);
                for (std::size_t i = e_begin; i < e_end; ++i) {
                    const auto e = kokkos_hg.vertex_edges(i);
                    const int lab = static_cast<int>(edge_labels(e));
                    if (lab >= 0 && lab < MAX_LABELS) counts[lab] += 1.0f;
                }

                int best = vertex_labels(v);
                float best_w = -1.0f;
                for (int lab = 0; lab < MAX_LABELS; ++lab) {
                    if (counts[lab] > best_w) {
                        best_w = counts[lab];
                        best = lab;
                    }
                }

                if (vertex_labels(v) != static_cast<Hypergraph::Label>(best)) {
                    vertex_labels(v) = static_cast<Hypergraph::Label>(best);
                    local_changes += 1;
                }
            },
            changes);

        const double change_ratio = static_cast<double>(changes) / static_cast<double>(kokkos_hg.num_vertices);
        if (change_ratio < tolerance) {
            std::cout << "Converged after " << iteration + 1 << " iterations\n";
            break;
        }
        if ((iteration + 1) % 10 == 0) {
            std::cout << "Iteration " << iteration + 1 << " completed\n";
        }
    }

    // Copy results back to host
    auto host_labels = Kokkos::create_mirror_view(vertex_labels);
    Kokkos::deep_copy(host_labels, vertex_labels);
    std::vector<Hypergraph::Label> final_labels(host_init_labels.size());
    for (std::size_t i = 0; i < host_init_labels.size(); ++i) {
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

// run_iteration_kokkos removed: algorithm is now implemented inline in run() with two phases
