#include "label_propagation_openmp.hpp"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iostream>

LabelPropagationOpenMP::LabelPropagationOpenMP(int num_threads) 
    : num_threads_(num_threads) {
    if (num_threads_ <= 0) {
        num_threads_ = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads_);
}

int LabelPropagationOpenMP::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running OpenMP label propagation with " << num_threads_ << " threads\n";
    
    std::size_t num_vertices = hypergraph.get_num_vertices();
    std::vector<Hypergraph::Label> current_labels = hypergraph.get_labels();
    std::vector<Hypergraph::Label> new_labels(num_vertices);

    int iteration = 0;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        // Parallel label update
        #pragma omp parallel for
        for (std::size_t v = 0; v < num_vertices; ++v) {
            new_labels[v] = compute_new_label(hypergraph, v);
        }

        // Check convergence
        if (check_convergence(current_labels, new_labels, tolerance)) {
            std::cout << "Converged after " << iteration + 1 << " iterations\n";
            break;
        }

        current_labels = new_labels;
        
        if ((iteration + 1) % 10 == 0) {
            std::cout << "Iteration " << iteration + 1 << " completed\n";
        }
    }

    hypergraph.set_labels(new_labels);
    return iteration + 1;
}

Hypergraph::Label LabelPropagationOpenMP::compute_new_label(const Hypergraph& hypergraph, 
                                                           Hypergraph::VertexId vertex_id) {
    const auto& labels = hypergraph.get_labels();
    const auto& incident_edges = hypergraph.get_incident_edges(vertex_id);
    
    if (incident_edges.empty()) {
        return labels[vertex_id];  // Keep current label if isolated
    }

    std::unordered_map<Hypergraph::Label, double> label_weights;

    // For each incident hyperedge, collect labels of neighbors
    for (auto edge_id : incident_edges) {
        const auto& edge_vertices = hypergraph.get_hyperedge(edge_id);
        std::size_t edge_size = edge_vertices.size();
        
        // Weight contribution is proportional to 1/edge_size
        double weight = 1.0 / static_cast<double>(edge_size);
        
        for (auto neighbor_id : edge_vertices) {
            if (neighbor_id != vertex_id) {
                label_weights[labels[neighbor_id]] += weight;
            }
        }
    }

    // Find the label with maximum weight
    Hypergraph::Label best_label = labels[vertex_id];
    double max_weight = 0.0;
    
    for (const auto& [label, weight] : label_weights) {
        if (weight > max_weight) {
            max_weight = weight;
            best_label = label;
        }
    }

    return best_label;
}

bool LabelPropagationOpenMP::check_convergence(const std::vector<Hypergraph::Label>& old_labels, 
                                              const std::vector<Hypergraph::Label>& new_labels,
                                              double tolerance) {
    std::size_t num_vertices = old_labels.size();
    std::size_t changes = 0;

    #pragma omp parallel for reduction(+:changes)
    for (std::size_t i = 0; i < num_vertices; ++i) {
        if (old_labels[i] != new_labels[i]) {
            changes++;
        }
    }

    double change_ratio = static_cast<double>(changes) / static_cast<double>(num_vertices);
    return change_ratio < tolerance;
}