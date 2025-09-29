#ifndef HYPERGRAPH_HPP
#define HYPERGRAPH_HPP

#include <vector>
#include <memory>
#include <cstddef>
#include <string>

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

private:
    std::size_t num_vertices_;
    std::vector<std::vector<VertexId>> hyperedges_;         // hyperedges_[e] = vertices in edge e
    std::vector<std::vector<EdgeId>> incident_edges_;       // incident_edges_[v] = edges incident to vertex v
    std::vector<Label> labels_;                             // labels_[v] = label of vertex v
    std::vector<std::size_t> degrees_;                      // degrees_[v] = degree of vertex v
    std::vector<std::size_t> edge_sizes_;                   // edge_sizes_[e] = size of edge e
};

/**
 * @brief Abstract base class for label propagation algorithms
 */
class LabelPropagationAlgorithm {
public:
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
};

#endif // HYPERGRAPH_HPP