#pragma once

#include "hypergraph.hpp"
#include <omp.h>
#include <string>

/**
 * @brief OpenMP implementation of hypergraph label propagation
 */
class LabelPropagationOpenMP : public LabelPropagationAlgorithm {
public:
    /**
     * @brief Constructor
     * @param num_threads Number of OpenMP threads to use
     */
    explicit LabelPropagationOpenMP(int num_threads = 0);

    /**
     * @brief Run the label propagation algorithm using OpenMP
     */
    int run(Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    /**
     * @brief Get the name of the implementation
     */
    std::string get_name() const override { return "OpenMP"; }

private:
    int num_threads_;
    
    /**
     * @brief Compute the most frequent label among neighbors
     */
    Hypergraph::Label compute_new_label(const Hypergraph& hypergraph, Hypergraph::VertexId vertex_id);
    
    /**
     * @brief Check convergence
     */
    bool check_convergence(const std::vector<Hypergraph::Label>& old_labels, 
                          const std::vector<Hypergraph::Label>& new_labels,
                          double tolerance);
};
