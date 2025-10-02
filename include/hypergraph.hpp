#pragma once

#include "argparse.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Hypergraph data structure
 *
 * A hypergraph H = (V, E) where V is a set of vertices and E is a set of hyperedges.
 * Each hyperedge can connect multiple vertices.
 */
class Hypergraph {
  public:
    using VertexId = std::size_t;
    using EdgeId = std::size_t;
    using Label = int;

    /**
     * @brief Flatten hypergraph data for GPU processing
     */
    struct FlatHypergraph {
        std::vector<Hypergraph::VertexId> edge_vertices; // Flattened vertex list
        std::vector<std::size_t> edge_offsets;           // Offsets into edge_vertices
        std::vector<Hypergraph::EdgeId> vertex_edges;    // Flattened edge list per vertex
        std::vector<std::size_t> vertex_offsets;         // Offsets into vertex_edges
        std::vector<std::size_t> edge_sizes;             // Size of each edge
        std::size_t num_vertices;
        std::size_t num_edges;
    };

    /**
     * @brief Constructor
     * @param num_vertices Number of vertices in the hypergraph
     */
    explicit Hypergraph(std::size_t num_vertices);

    /**
     * @brief Add a hyperedge to the graph
     * @param vertices Vector of vertex IDs that form the hyperedge
     * @return The ID of the created hyperedge
     */
    EdgeId add_hyperedge(const std::vector<VertexId>& vertices);

    /**
     * @brief Get the number of vertices
     */
    std::size_t get_num_vertices() const { return num_vertices_; }

    /**
     * @brief Get the number of hyperedges
     */
    std::size_t get_num_edges() const { return hyperedges_.size(); }

    /**
     * @brief Get vertices in a hyperedge
     */
    const std::vector<VertexId>& get_hyperedge(EdgeId edge_id) const;

    /**
     * @brief Get hyperedges incident to a vertex
     */
    const std::vector<EdgeId>& get_incident_edges(VertexId vertex_id) const;

    /**
     * @brief Get vertex labels
     */
    const std::vector<Label>& get_labels() const { return labels_; }

    /**
     * @brief Set vertex labels
     */
    void set_labels(const std::vector<Label>& labels);

    /**
     * @brief Get vertex degrees (number of incident hyperedges)
     */
    const std::vector<std::size_t>& get_degrees() const { return degrees_; }

    /**
     * @brief Get hyperedge sizes
     */
    const std::vector<std::size_t>& get_edge_sizes() const { return edge_sizes_; }

    /**
     * @brief Flatten hypergraph data for GPU processing
     */
    FlatHypergraph flatten() const;

    // ---------------------------
    // Binary serialization
    // ---------------------------
    // Format v1 (little-endian):
    //   uint32 magic = 'HGR1'
    //   uint32 version = 1
    //   uint64 num_vertices
    //   uint64 num_edges
    //   repeat num_edges times:
    //       uint64 edge_size
    //       uint64 vertices[edge_size]
    //   uint8  has_labels (0 or 1)
    //   if has_labels:
    //       int32 labels[num_vertices]
    void save_to_file(const std::string& path) const;
    static std::unique_ptr<Hypergraph> load_from_file(const std::string& path);

  private:
    std::size_t num_vertices_;
    std::vector<std::vector<VertexId>> hyperedges_;   // hyperedges_[e] = vertices in edge e
    std::vector<std::vector<EdgeId>> incident_edges_; // incident_edges_[v] = edges incident to vertex v
    std::vector<Label> labels_;                       // labels_[v] = label of vertex v
    std::vector<std::size_t> degrees_;                // degrees_[v] = degree of vertex v
    std::vector<std::size_t> edge_sizes_;             // edge_sizes_[e] = size of edge e
};

/**
 * @brief Abstract base class for label propagation algorithms
 */
class LabelPropagationAlgorithm {
  public:
    explicit LabelPropagationAlgorithm(const CLI::DeviceOptions& device) : device_(device) {}
    virtual ~LabelPropagationAlgorithm() = default;

    /**
     * @brief Run the label propagation algorithm
     * @param hypergraph The input hypergraph
     * @param max_iterations Maximum number of iterations
     * @param tolerance Convergence tolerance
     * @return Number of iterations performed
     */
    virtual int run(Hypergraph& hypergraph, int max_iterations = 100, double tolerance = 1e-6) = 0;

    /**
     * @brief Get the name of the implementation
     */
    virtual std::string get_name() const = 0;

  protected:
    CLI::DeviceOptions device_;
};

/**
 * @brief Random hypergraph generators (common library)
 */
namespace hypergraph_generators {

/**
 * @brief Generate a random hypergraph with uniformly distributed edge sizes.
 * @param num_vertices Number of vertices
 * @param num_edges Number of hyperedges
 * @param min_edge_size Minimum edge cardinality (>=2)
 * @param max_edge_size Maximum edge cardinality (>=min_edge_size)
 * @param seed RNG seed (optional; if 0 uses nondeterministic seed)
 */
std::unique_ptr<Hypergraph> generate_uniform(std::size_t num_vertices, std::size_t num_edges, std::size_t min_edge_size = 2, std::size_t max_edge_size = 5, unsigned int seed = 0);

/**
 * @brief Generate a random hypergraph where all edges have the same size.
 * @param num_vertices Number of vertices
 * @param num_edges Number of hyperedges
 * @param edge_size Fixed edge cardinality (>=2)
 * @param seed RNG seed (optional; if 0 uses nondeterministic seed)
 */
std::unique_ptr<Hypergraph> generate_fixed_edge_size(std::size_t num_vertices, std::size_t num_edges, std::size_t edge_size, unsigned int seed = 0);


/**
 * @brief Generate a planted-partition (community) hypergraph.
 *        Edges are intra-community with probability p_intra, otherwise inter-community.
 * @param num_vertices Number of vertices
 * @param num_edges Number of hyperedges
 * @param num_communities Number of communities (>=1)
 * @param p_intra Probability an edge is intra-community [0,1]
 * @param min_edge_size Minimum edge size
 * @param max_edge_size Maximum edge size
 * @param seed RNG seed (optional; if 0 uses nondeterministic seed)
 */
std::unique_ptr<Hypergraph> generate_planted_partition(
    std::size_t num_vertices, std::size_t num_edges, std::size_t num_communities, double p_intra = 0.8, std::size_t min_edge_size = 2, std::size_t max_edge_size = 5, unsigned int seed = 0);

/**
 * @brief Generate a hypergraph Stochastic Block Model (hSBM) with two edge types:
 *        intra-community edges vs mixed-community edges.
 *        For each edge, sample size k in [min_edge_size, max_edge_size].
 *        Then sample a candidate k-set of vertices uniformly at random.
 *        Accept with probability p_intra if all vertices in the set belong to the same community,
 *        otherwise accept with probability p_inter. Repeat until num_edges edges are accepted.
 */
std::unique_ptr<Hypergraph> generate_hsbm(
    std::size_t num_vertices, std::size_t num_edges, std::size_t num_communities, double p_intra, double p_inter, std::size_t min_edge_size = 2, std::size_t max_edge_size = 5, unsigned int seed = 0);

/**
 * @brief Generate random labels for vertices in range [0, num_classes).
 * @param num_vertices Number of vertices
 * @param num_classes Number of distinct labels/classes (>=1)
 * @param seed RNG seed (optional; if 0 uses nondeterministic seed)
 */
std::vector<Hypergraph::Label> generate_random_labels(std::size_t num_vertices, std::size_t num_classes, unsigned int seed = 0);

} // namespace hypergraph_generators
