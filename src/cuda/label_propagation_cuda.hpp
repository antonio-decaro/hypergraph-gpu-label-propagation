#pragma once

#include "hypergraph.hpp"
#include <cuda_runtime.h>
#include <string>

/**
 * @brief CUDA implementation of hypergraph label propagation
 */
class LabelPropagationCUDA : public LabelPropagationAlgorithm {
  public:
    explicit LabelPropagationCUDA(const CLI::DeviceOptions& device);
    ~LabelPropagationCUDA();

    PerformanceMeasurer run(Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    std::string get_name() const override { return "CUDA"; }

  private:
    struct DeviceFlatHypergraph {
        Hypergraph::VertexId* edge_vertices = nullptr;
        std::size_t* edge_offsets = nullptr;
        Hypergraph::EdgeId* vertex_edges = nullptr;
        std::size_t* vertex_offsets = nullptr;
        std::size_t* edge_sizes = nullptr;
        std::size_t num_vertices = 0;
        std::size_t num_edges = 0;
    };

    DeviceFlatHypergraph create_device_hypergraph(const Hypergraph& hypergraph);
    void destroy_device_hypergraph(DeviceFlatHypergraph& flat_hg);
    bool run_iteration_cuda(const DeviceFlatHypergraph& flat_hg,
                            Hypergraph::Label* d_vertex_labels,
                            Hypergraph::Label* d_edge_labels,
                            unsigned int* d_change_flags,
                            double tolerance);

    static void check_cuda(cudaError_t err, const char* context);

    int device_id_ = 0;
    int max_threads_per_block_ = 1024;
};
