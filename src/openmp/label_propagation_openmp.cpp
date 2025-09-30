#include "label_propagation_openmp.hpp"
#include <unordered_map>
#include <algorithm>
#include <numeric>
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
    std::cout << "Running OpenMP target (GPU) label propagation with " << num_threads_ << " threads\n";

    const std::size_t num_vertices = hypergraph.get_num_vertices();
    const std::size_t num_edges = hypergraph.get_num_edges();

    // Flatten hypergraph for device-friendly access
    Hypergraph::FlatHypergraph flat = hypergraph.flatten();

    // Host-side label buffers
    std::vector<Hypergraph::Label> current_labels = hypergraph.get_labels();
    std::vector<Hypergraph::Label> new_labels(num_vertices, 0);

    // Precompute per-vertex neighbor counts: sum over incident edges of (edge_size - 1)
    std::vector<std::size_t> neighbor_counts(num_vertices, 0);
    for (std::size_t v = 0; v < num_vertices; ++v) {
        const std::size_t ve_begin = flat.vertex_offsets[v];
        const std::size_t ve_end = flat.vertex_offsets[v + 1];
        std::size_t count = 0;
        for (std::size_t i = ve_begin; i < ve_end; ++i) {
            const auto e = flat.vertex_edges[i];
            count += (flat.edge_sizes[e] > 0 ? flat.edge_sizes[e] - 1 : 0);
        }
        neighbor_counts[v] = count;
    }

    // Prefix sums for per-vertex scratch segments
    std::vector<std::size_t> scratch_offsets(num_vertices + 1, 0);
    for (std::size_t v = 0; v < num_vertices; ++v) {
        scratch_offsets[v + 1] = scratch_offsets[v] + neighbor_counts[v];
    }
    const std::size_t scratch_size = scratch_offsets.back();

    // Scratch storage for per-vertex temporary label aggregation
    std::vector<int> scratch_labels(scratch_size, 0);
    std::vector<double> scratch_weights(scratch_size, 0.0);

    // Cache sizes used in OpenMP map clauses
    const std::size_t edge_vertices_size = flat.edge_vertices.size();
    const std::size_t edge_offsets_size = flat.edge_offsets.size();
    const std::size_t vertex_edges_size = flat.vertex_edges.size();
    const std::size_t vertex_offsets_size = flat.vertex_offsets.size();

    int iteration = 0;
    bool converged = false;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        // Update pointers to host buffers for mapping
        Hypergraph::Label* current_labels_ptr = current_labels.data();
        Hypergraph::Label* new_labels_ptr = new_labels.data();
        const Hypergraph::VertexId* edge_vertices = flat.edge_vertices.data();
        const std::size_t* edge_offsets = flat.edge_offsets.data();
        const Hypergraph::EdgeId* vertex_edges = flat.vertex_edges.data();
        const std::size_t* vertex_offsets = flat.vertex_offsets.data();
        const std::size_t* edge_sizes = flat.edge_sizes.data();
        const std::size_t* scratch_offsets_ptr = scratch_offsets.data();
        int* scratch_labels_ptr = scratch_labels.data();
        double* scratch_weights_ptr = scratch_weights.data();
        
        // Compute new labels on the device using OpenMP target offload
        #pragma omp target teams distribute parallel for \
            map(to: \
                current_labels_ptr[0:num_vertices], \
                edge_vertices[0:edge_vertices_size], \
                edge_offsets[0:edge_offsets_size], \
                vertex_edges[0:vertex_edges_size], \
                vertex_offsets[0:vertex_offsets_size], \
                edge_sizes[0:num_edges], \
                scratch_offsets_ptr[0:(num_vertices+1)] \
            ) \
            map(from: new_labels_ptr[0:num_vertices]) \
            map(alloc: scratch_labels_ptr[0:scratch_size], scratch_weights_ptr[0:scratch_size])
        for (std::size_t v = 0; v < num_vertices; ++v) {
            const std::size_t seg_begin = scratch_offsets_ptr[v];
            std::size_t used = 0; // number of unique labels stored in this segment

            const std::size_t ve_begin = vertex_offsets[v];
            const std::size_t ve_end = vertex_offsets[v + 1];

            // Collect neighbor labels with weights into the per-vertex scratch segment
            for (std::size_t i = ve_begin; i < ve_end; ++i) {
                const auto e = vertex_edges[i];
                const std::size_t ev_begin = edge_offsets[e];
                const std::size_t ev_end = edge_offsets[e + 1];
                const std::size_t esize = edge_sizes[e];
                const double w = esize > 0 ? (1.0 / static_cast<double>(esize)) : 0.0;

                for (std::size_t j = ev_begin; j < ev_end; ++j) {
                    const auto u = edge_vertices[j];
                    if (u == v) continue;
                    const int lab = current_labels_ptr[u];

                    // Linear probe for existing label in this vertex's segment
                    std::size_t k = 0;
                    for (; k < used; ++k) {
                        if (scratch_labels_ptr[seg_begin + k] == lab) {
                            scratch_weights_ptr[seg_begin + k] += w;
                            break;
                        }
                    }
                    if (k == used) {
                        // New label entry
                        scratch_labels_ptr[seg_begin + used] = lab;
                        scratch_weights_ptr[seg_begin + used] = w;
                        ++used;
                    }
                }
            }

            // Pick label with maximum accumulated weight
            double max_w = -1.0;
            int best_label = current_labels_ptr[v];
            for (std::size_t k = 0; k < used; ++k) {
                const double w = scratch_weights_ptr[seg_begin + k];
                if (w > max_w) {
                    max_w = w;
                    best_label = scratch_labels_ptr[seg_begin + k];
                }
            }
            new_labels_ptr[v] = best_label;

            // Optional: no need to clear weights; we overwrite used entries next iteration
        }

        // Convergence check on device: count label changes
        std::size_t changes = 0;
        // Refresh pointers (new_labels/current_labels may be reallocated if swapped)
        current_labels_ptr = current_labels.data();
        new_labels_ptr = new_labels.data();

        #pragma omp target teams distribute parallel for reduction(+:changes) \
            map(to: current_labels_ptr[0:num_vertices], new_labels_ptr[0:num_vertices])
        for (std::size_t i = 0; i < num_vertices; ++i) {
            if (current_labels_ptr[i] != new_labels_ptr[i]) {
                changes += 1;
            }
        }

        const double change_ratio = static_cast<double>(changes) / static_cast<double>(num_vertices);
        if (change_ratio < tolerance) {
            std::cout << "Converged after " << iteration + 1 << " iterations\n";
            converged = true;
            break;
        }

        current_labels.swap(new_labels);
        if ((iteration + 1) % 10 == 0) {
            std::cout << "Iteration " << iteration + 1 << " completed\n";
        }
    }

    // Set final labels
    // If converged, new_labels contains the latest computed values.
    // If not converged (hit max_iterations), current_labels was swapped to hold the latest values.
    if (!converged) {
        new_labels = current_labels;
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
