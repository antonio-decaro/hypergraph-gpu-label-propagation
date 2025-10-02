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
     * @param device Device options (number of threads, etc.)
     */
    explicit LabelPropagationOpenMP(const CLI::DeviceOptions& device);

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
};
