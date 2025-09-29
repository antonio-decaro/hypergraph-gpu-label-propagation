#ifndef LABEL_PROPAGATION_KOKKOS_HPP
#define LABEL_PROPAGATION_KOKKOS_HPP

#include "hypergraph.hpp"
#include <Kokkos_Core.hpp>
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
    LabelPropagationKokkos();

    /**
     * @brief Destructor
     */
    ~LabelPropagationKokkos();

    /**
     * @brief Run the label propagation algorithm using Kokkos
     */
    int run(Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    /**
     * @brief Get the name of the implementation
     */
    std::string get_name() const override { return "Kokkos"; }

private:
    bool kokkos_initialized_;

    /**
     * @brief Flattened hypergraph representation for Kokkos
     */
    struct KokkosHypergraph {
        VertexView edge_vertices;     // Flattened vertex list
        SizeView edge_offsets;        // Offsets into edge_vertices
        EdgeView vertex_edges;        // Flattened edge list per vertex
        SizeView vertex_offsets;      // Offsets into vertex_edges
        SizeView edge_sizes;          // Size of each edge
        std::size_t num_vertices;
        std::size_t num_edges;
    };
    
    /**
     * @brief Convert hypergraph to Kokkos representation
     */
    KokkosHypergraph create_kokkos_hypergraph(const Hypergraph& hypergraph);
    
    /**
     * @brief Run one iteration of label propagation
     */
    bool run_iteration_kokkos(const KokkosHypergraph& kokkos_hg,
                             const LabelView& current_labels,
                             const LabelView& new_labels,
                             double tolerance);
};

#endif // LABEL_PROPAGATION_KOKKOS_HPP