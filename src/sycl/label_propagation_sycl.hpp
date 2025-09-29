#ifndef LABEL_PROPAGATION_SYCL_HPP
#define LABEL_PROPAGATION_SYCL_HPP

#include "hypergraph.hpp"
#include <sycl/sycl.hpp>
#include <string>

/**
 * @brief SYCL implementation of hypergraph label propagation
 */
class LabelPropagationSYCL : public LabelPropagationAlgorithm {
public:
    /**
     * @brief Constructor
     * @param device_selector SYCL device selector (default: GPU, fallback to CPU)
     */
    explicit LabelPropagationSYCL(sycl::device_selector* device_selector = nullptr);

    /**
     * @brief Destructor
     */
    ~LabelPropagationSYCL();

    /**
     * @brief Run the label propagation algorithm using SYCL
     */
    int run(Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    /**
     * @brief Get the name of the implementation
     */
    std::string get_name() const override { return "SYCL"; }

private:
    sycl::queue queue_;
    sycl::device device_;
    
    /**
     * @brief Flatten hypergraph data for GPU processing
     */
    struct FlatHypergraph {
        std::vector<Hypergraph::VertexId> edge_vertices;    // Flattened vertex list
        std::vector<std::size_t> edge_offsets;              // Offsets into edge_vertices
        std::vector<Hypergraph::EdgeId> vertex_edges;       // Flattened edge list per vertex
        std::vector<std::size_t> vertex_offsets;            // Offsets into vertex_edges
        std::vector<std::size_t> edge_sizes;                // Size of each edge
        std::size_t num_vertices;
        std::size_t num_edges;
    };
    
    /**
     * @brief Convert hypergraph to flat representation
     */
    FlatHypergraph flatten_hypergraph(const Hypergraph& hypergraph);
    
    /**
     * @brief Run one iteration of label propagation on GPU
     */
    bool run_iteration_sycl(const FlatHypergraph& flat_hg,
                           sycl::buffer<Hypergraph::Label, 1>& current_labels,
                           sycl::buffer<Hypergraph::Label, 1>& new_labels,
                           double tolerance);
};

#endif // LABEL_PROPAGATION_SYCL_HPP