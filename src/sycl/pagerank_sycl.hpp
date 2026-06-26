#pragma once

#include "hypergraph.hpp"
#include <string>
#include <sycl/sycl.hpp>

class PageRankSYCL : public PageRankAlgorithm {
  public:
    explicit PageRankSYCL(const CLI::DeviceOptions& device, const sycl::queue& queue, double alpha = 0.85);
    ~PageRankSYCL();

    PerformanceMeasurer run(const Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    std::string get_name() const override { return "SYCL-PageRank"; }

  private:
    sycl::queue queue_;

    struct DeviceFlatHypergraph {
        Hypergraph::VertexId* edge_vertices = nullptr;
        std::size_t* edge_offsets           = nullptr;
        Hypergraph::EdgeId* vertex_edges    = nullptr;
        std::size_t* vertex_offsets         = nullptr;
        std::size_t* edge_sizes             = nullptr;
        std::size_t num_vertices = 0;
        std::size_t num_edges    = 0;
    };

    DeviceFlatHypergraph create_device_hypergraph(const Hypergraph& hypergraph) {
        auto flat_hg = hypergraph.flatten();
        DeviceFlatHypergraph dh;
        dh.edge_vertices  = sycl::malloc_device<Hypergraph::VertexId>(flat_hg.edge_vertices.size(), queue_);
        dh.edge_offsets   = sycl::malloc_device<std::size_t>(flat_hg.edge_offsets.size(), queue_);
        dh.vertex_edges   = sycl::malloc_device<Hypergraph::EdgeId>(flat_hg.vertex_edges.size(), queue_);
        dh.vertex_offsets = sycl::malloc_device<std::size_t>(flat_hg.vertex_offsets.size(), queue_);
        dh.edge_sizes     = sycl::malloc_device<std::size_t>(flat_hg.edge_sizes.size(), queue_);
        dh.num_vertices   = flat_hg.num_vertices;
        dh.num_edges      = flat_hg.num_edges;
        queue_.copy(flat_hg.edge_vertices.data(),  dh.edge_vertices,  flat_hg.edge_vertices.size());
        queue_.copy(flat_hg.edge_offsets.data(),   dh.edge_offsets,   flat_hg.edge_offsets.size());
        queue_.copy(flat_hg.vertex_edges.data(),   dh.vertex_edges,   flat_hg.vertex_edges.size());
        queue_.copy(flat_hg.vertex_offsets.data(), dh.vertex_offsets, flat_hg.vertex_offsets.size());
        queue_.copy(flat_hg.edge_sizes.data(),     dh.edge_sizes,     flat_hg.edge_sizes.size());
        queue_.wait();
        return dh;
    }

    void cleanup_flat_hypergraph(DeviceFlatHypergraph& dh) {
        auto safe_free = [&](auto*& ptr) { if (ptr) { sycl::free(ptr, queue_); ptr = nullptr; } };
        safe_free(dh.edge_vertices);
        safe_free(dh.edge_offsets);
        safe_free(dh.vertex_edges);
        safe_free(dh.vertex_offsets);
        safe_free(dh.edge_sizes);
    }

    bool run_iteration_sycl(const DeviceFlatHypergraph& flat_hg,
                            float* d_p, float* d_h, float* d_diff, double tolerance);
};
