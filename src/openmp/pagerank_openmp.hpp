#pragma once

#include "hypergraph.hpp"
#include <omp.h>
#include <string>

class PageRankOpenMP : public PageRankAlgorithm {
  public:
    explicit PageRankOpenMP(const CLI::DeviceOptions& device, double alpha = 0.85);

    PerformanceMeasurer run(const Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    std::string get_name() const override { return "OpenMP-PageRank"; }

  private:
    int num_threads_;
};
