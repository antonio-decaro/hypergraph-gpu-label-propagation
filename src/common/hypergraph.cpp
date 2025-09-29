#include "hypergraph.hpp"
#include <stdexcept>
#include <algorithm>

Hypergraph::Hypergraph(std::size_t num_vertices) 
    : num_vertices_(num_vertices)
    , incident_edges_(num_vertices)
    , labels_(num_vertices, 0)
    , degrees_(num_vertices, 0) {
}

Hypergraph::EdgeId Hypergraph::add_hyperedge(const std::vector<VertexId>& vertices) {
    if (vertices.empty()) {
        throw std::invalid_argument("Hyperedge cannot be empty");
    }
    
    for (VertexId v : vertices) {
        if (v >= num_vertices_) {
            throw std::invalid_argument("Vertex ID out of range");
        }
    }

    EdgeId edge_id = hyperedges_.size();
    hyperedges_.push_back(vertices);
    edge_sizes_.push_back(vertices.size());

    // Update incident edges and degrees
    for (VertexId v : vertices) {
        incident_edges_[v].push_back(edge_id);
        degrees_[v]++;
    }

    return edge_id;
}

const std::vector<Hypergraph::VertexId>& Hypergraph::get_hyperedge(EdgeId edge_id) const {
    if (edge_id >= hyperedges_.size()) {
        throw std::invalid_argument("Edge ID out of range");
    }
    return hyperedges_[edge_id];
}

const std::vector<Hypergraph::EdgeId>& Hypergraph::get_incident_edges(VertexId vertex_id) const {
    if (vertex_id >= num_vertices_) {
        throw std::invalid_argument("Vertex ID out of range");
    }
    return incident_edges_[vertex_id];
}

void Hypergraph::set_labels(const std::vector<Label>& labels) {
    if (labels.size() != num_vertices_) {
        throw std::invalid_argument("Labels size must match number of vertices");
    }
    labels_ = labels;
}