#pragma once

#include "hypergraph.hpp"
#include <Kokkos_Core.hpp>
#include <string>

class PageRankKokkos : public PageRankAlgorithm {
  public:
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace    = typename ExecutionSpace::memory_space;
    using FloatView      = Kokkos::View<float*, MemorySpace>;
    using VertexView     = Kokkos::View<Hypergraph::VertexId*, MemorySpace>;
    using EdgeView       = Kokkos::View<Hypergraph::EdgeId*, MemorySpace>;
    using SizeView       = Kokkos::View<std::size_t*, MemorySpace>;

    explicit PageRankKokkos(const CLI::DeviceOptions& device, double alpha = 0.85);
    ~PageRankKokkos();

    PerformanceMeasurer run(const Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    std::string get_name() const override { return "Kokkos-PageRank"; }

    struct KokkosHypergraph {
        VertexView edge_vertices;
        SizeView   edge_offsets;
        EdgeView   vertex_edges;
        SizeView   vertex_offsets;
        SizeView   edge_sizes;
        std::size_t num_vertices = 0;
        std::size_t num_edges    = 0;
    };

  private:
    bool kokkos_initialized_;
    KokkosHypergraph create_kokkos_hypergraph(const Hypergraph& hypergraph);
};
