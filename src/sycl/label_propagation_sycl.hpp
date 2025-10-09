#pragma once

#include "hypergraph.hpp"
#include <string>
#include <sycl/sycl.hpp>

/**
 * @brief SYCL implementation of hypergraph label propagation
 */
class LabelPropagationSYCL : public LabelPropagationAlgorithm {
  public:
    /**
     * @brief Constructor
     * @param queue SYCL queue for device execution
     */
    explicit LabelPropagationSYCL(const CLI::DeviceOptions& device, const sycl::queue& queue);

    /**
     * @brief Destructor
     */
    ~LabelPropagationSYCL();

    /**
     * @brief Run the label propagation algorithm using SYCL
     */
    PerformanceMeasurer run(Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) override;

    /**
     * @brief Get the name of the implementation
     */
    std::string get_name() const override { return "SYCL"; }

  private:
    sycl::queue queue_;

    /**
     * @brief Flatten hypergraph data for GPU processing
     */
    struct DeviceFlatHypergraph {
        Hypergraph::VertexId* edge_vertices; // Flattened vertex list
        std::size_t* edge_offsets;           // Offsets into edge_vertices
        Hypergraph::EdgeId* vertex_edges;    // Flattened edge list per vertex
        std::size_t* vertex_offsets;         // Offsets into vertex_edges
        std::size_t* edge_sizes;             // Size of each edge
        std::size_t num_vertices;
        std::size_t num_edges;
    };

    /**
     * @brief Convert hypergraph to flat representation
     */
    DeviceFlatHypergraph create_device_hypergraph(const Hypergraph& hypergraph) {
        auto flat_hg = hypergraph.flatten();

        // Allocate USM memory for flat structures
        DeviceFlatHypergraph device_flat_hg;
        device_flat_hg.edge_vertices = sycl::malloc_device<Hypergraph::VertexId>(flat_hg.edge_vertices.size(), queue_);
        device_flat_hg.edge_offsets = sycl::malloc_device<std::size_t>(flat_hg.edge_offsets.size(), queue_);
        device_flat_hg.vertex_edges = sycl::malloc_device<Hypergraph::EdgeId>(flat_hg.vertex_edges.size(), queue_);
        device_flat_hg.vertex_offsets = sycl::malloc_device<std::size_t>(flat_hg.vertex_offsets.size(), queue_);
        device_flat_hg.edge_sizes = sycl::malloc_device<std::size_t>(flat_hg.edge_sizes.size(), queue_);
        device_flat_hg.num_vertices = flat_hg.num_vertices;
        device_flat_hg.num_edges = flat_hg.num_edges;

        // Copy data to device
        queue_.copy(flat_hg.edge_vertices.data(), device_flat_hg.edge_vertices, flat_hg.edge_vertices.size());
        queue_.copy(flat_hg.edge_offsets.data(), device_flat_hg.edge_offsets, flat_hg.edge_offsets.size());
        queue_.copy(flat_hg.vertex_edges.data(), device_flat_hg.vertex_edges, flat_hg.vertex_edges.size());
        queue_.copy(flat_hg.vertex_offsets.data(), device_flat_hg.vertex_offsets, flat_hg.vertex_offsets.size());
        queue_.copy(flat_hg.edge_sizes.data(), device_flat_hg.edge_sizes, flat_hg.edge_sizes.size());
        queue_.wait();

        return device_flat_hg;
    }

    void cleanup_flat_hypergraph(DeviceFlatHypergraph& flat_hg) {
        if (flat_hg.edge_vertices) {
            sycl::free(flat_hg.edge_vertices, queue_);
            flat_hg.edge_vertices = nullptr;
        }
        if (flat_hg.edge_offsets) {
            sycl::free(flat_hg.edge_offsets, queue_);
            flat_hg.edge_offsets = nullptr;
        }
        if (flat_hg.vertex_edges) {
            sycl::free(flat_hg.vertex_edges, queue_);
            flat_hg.vertex_edges = nullptr;
        }
        if (flat_hg.vertex_offsets) {
            sycl::free(flat_hg.vertex_offsets, queue_);
            flat_hg.vertex_offsets = nullptr;
        }
        if (flat_hg.edge_sizes) {
            sycl::free(flat_hg.edge_sizes, queue_);
            flat_hg.edge_sizes = nullptr;
        }
    }

    /**
     * @brief Run one iteration of label propagation on GPU
     */
    bool run_iteration_sycl(const DeviceFlatHypergraph& flat_hg,
        Hypergraph::Label* vertex_labels,
        Hypergraph::Label* edge_labels,
        std::size_t* changes,
        int max_labels,
        double tolerance);
};
