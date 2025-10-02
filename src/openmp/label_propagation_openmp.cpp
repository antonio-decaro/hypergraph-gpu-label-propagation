#include "label_propagation_openmp.hpp"
#include <algorithm>
#include <iostream>

LabelPropagationOpenMP::LabelPropagationOpenMP(CLI::DeviceOptions device) : LabelPropagationAlgorithm(device) {
    num_threads_ = static_cast<int>(device_.threads);
    if (num_threads_ <= 0) { num_threads_ = omp_get_max_threads(); }
    omp_set_num_threads(num_threads_);
}

int LabelPropagationOpenMP::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running OpenMP target (GPU) label propagation with " << num_threads_ << " threads\n";

    const std::size_t num_vertices = hypergraph.get_num_vertices();
    const std::size_t num_edges = hypergraph.get_num_edges();

    // Flatten hypergraph for device-friendly access
    Hypergraph::FlatHypergraph flat = hypergraph.flatten();

    // Host-side labels: vertex labels initialized from input; edge labels start at 0
    std::vector<Hypergraph::Label> vertex_labels = hypergraph.get_labels();
    std::vector<Hypergraph::Label> edge_labels(num_edges, 0);

    // Cache sizes used in OpenMP map clauses
    const std::size_t edge_vertices_size = flat.edge_vertices.size();
    const std::size_t edge_offsets_size = flat.edge_offsets.size();
    const std::size_t vertex_edges_size = flat.vertex_edges.size();
    const std::size_t vertex_offsets_size = flat.vertex_offsets.size();

    // SYCL version assumes small bounded label space; mirror that here
    constexpr int MAX_LABELS = 10; // must be >= number of possible labels

    int iteration = 0;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        Hypergraph::Label* vlabels = vertex_labels.data();
        Hypergraph::Label* elabels = edge_labels.data();
        const Hypergraph::VertexId* edge_vertices = flat.edge_vertices.data();
        const std::size_t* edge_offsets = flat.edge_offsets.data();
        const Hypergraph::EdgeId* vertex_edges = flat.vertex_edges.data();
        const std::size_t* vertex_offsets = flat.vertex_offsets.data();

// Phase 1: update edge labels from incident vertex labels (unweighted counts)
#pragma omp target teams distribute parallel for map(to : vlabels[0 : num_vertices], edge_vertices[0 : edge_vertices_size], edge_offsets[0 : edge_offsets_size]) map(tofrom : elabels[0 : num_edges])
        for (std::size_t e = 0; e < num_edges; ++e) {
            float counts[MAX_LABELS];
            for (int i = 0; i < MAX_LABELS; ++i) counts[i] = 0.0f;

            const std::size_t v_begin = edge_offsets[e];
            const std::size_t v_end = edge_offsets[e + 1];
            for (std::size_t j = v_begin; j < v_end; ++j) {
                const auto u = edge_vertices[j];
                const int lab = static_cast<int>(vlabels[u]);
                if (lab >= 0 && lab < MAX_LABELS) counts[lab] += 1.0f;
            }

            int best = elabels[e];
            float best_w = -1.0f;
            for (int lab = 0; lab < MAX_LABELS; ++lab) {
                if (counts[lab] > best_w) {
                    best_w = counts[lab];
                    best = lab;
                }
            }
            elabels[e] = static_cast<Hypergraph::Label>(best);
        }

        // Phase 2: update vertex labels from incident edge labels; count changes
        std::size_t changes = 0;
#pragma omp target teams distribute parallel for reduction(+ : changes) map(to : elabels[0 : num_edges], vertex_edges[0 : vertex_edges_size], vertex_offsets[0 : vertex_offsets_size])                 \
    map(tofrom : vlabels[0 : num_vertices])
        for (std::size_t v = 0; v < num_vertices; ++v) {
            float counts[MAX_LABELS];
            for (int i = 0; i < MAX_LABELS; ++i) counts[i] = 0.0f;

            const std::size_t e_begin = vertex_offsets[v];
            const std::size_t e_end = vertex_offsets[v + 1];
            for (std::size_t i = e_begin; i < e_end; ++i) {
                const auto e = vertex_edges[i];
                const int lab = static_cast<int>(elabels[e]);
                if (lab >= 0 && lab < MAX_LABELS) counts[lab] += 1.0f;
            }

            int best = vlabels[v];
            float best_w = -1.0f;
            for (int lab = 0; lab < MAX_LABELS; ++lab) {
                if (counts[lab] > best_w) {
                    best_w = counts[lab];
                    best = lab;
                }
            }

            if (vlabels[v] != static_cast<Hypergraph::Label>(best)) {
                vlabels[v] = static_cast<Hypergraph::Label>(best);
                changes += 1;
            }
        }

        const double change_ratio = static_cast<double>(changes) / static_cast<double>(num_vertices);
        if (change_ratio < tolerance) {
            std::cout << "Converged after " << iteration + 1 << " iterations\n";
            break;
        }
        if ((iteration + 1) % 10 == 0) { std::cout << "Iteration " << iteration + 1 << " completed\n"; }
    }

    hypergraph.set_labels(vertex_labels);
    return iteration + 1;
}
