#pragma once

#include "hypergraph.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <string>

/**
 * @brief Kokkos implementation of hypergraph label propagation
 */
class LabelPropagationKokkos : public LabelPropagationAlgorithm {
  public:
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;

    // Kokkos Views for GPU-friendly data structures
    using LabelView = Kokkos::View<Hypergraph::Label*, MemorySpace>;
    using VertexView = Kokkos::View<Hypergraph::VertexId*, MemorySpace>;
    using EdgeView = Kokkos::View<Hypergraph::EdgeId*, MemorySpace>;
    using SizeView = Kokkos::View<std::size_t*, MemorySpace>;
    using CounterView = Kokkos::View<std::size_t*, MemorySpace>;

    /**
     * @brief Constructor
     */
    LabelPropagationKokkos(const CLI::DeviceOptions& device);

    /**
     * @brief Destructor
     */
    ~LabelPropagationKokkos();

    /**
     * @brief Run the label propagation algorithm using Kokkos
     */
    PerformanceMeasurer run(Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    /**
     * @brief Get the name of the implementation
     */
    std::string get_name() const override { return "Kokkos"; }

    /**
     * @brief Flattened hypergraph representation for Kokkos
     *        Needs to be public so it can be captured in device lambdas.
     */
    struct KokkosHypergraph {
        VertexView edge_vertices; // Flattened vertex list
        SizeView edge_offsets;    // Offsets into edge_vertices
        EdgeView vertex_edges;    // Flattened edge list per vertex
        SizeView vertex_offsets;  // Offsets into vertex_edges
        SizeView edge_sizes;      // Size of each edge
        std::size_t num_vertices;
        std::size_t num_edges;
    };

    struct ExecutionPool {
        Kokkos::View<std::uint32_t*, MemorySpace> wg_pool_edges;
        std::size_t wg_pool_edges_size{0};
        Kokkos::View<std::uint32_t*, MemorySpace> sg_pool_edges;
        std::size_t sg_pool_edges_size{0};
        Kokkos::View<std::uint32_t*, MemorySpace> wi_pool_edges;
        std::size_t wi_pool_edges_size{0};

        Kokkos::View<std::uint32_t*, MemorySpace> wg_pool_vertices;
        std::size_t wg_pool_vertices_size{0};
        Kokkos::View<std::uint32_t*, MemorySpace> sg_pool_vertices;
        std::size_t sg_pool_vertices_size{0};
        Kokkos::View<std::uint32_t*, MemorySpace> wi_pool_vertices;
        std::size_t wi_pool_vertices_size{0};
    };

  private:
    bool kokkos_initialized_;

    /**
     * @brief Convert hypergraph to Kokkos representation
     */
    KokkosHypergraph create_kokkos_hypergraph(const Hypergraph& hypergraph);

    ExecutionPool create_execution_pool(const Hypergraph& hypergraph);
};
