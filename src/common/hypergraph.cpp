#include "hypergraph.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <istream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

Hypergraph::Hypergraph(std::size_t num_vertices) : num_vertices_(num_vertices), incident_edges_(num_vertices), labels_(num_vertices, 0), degrees_(num_vertices, 0) {}

Hypergraph::EdgeId Hypergraph::add_hyperedge(const std::vector<VertexId>& vertices) {
    if (vertices.empty()) { throw std::invalid_argument("Hyperedge cannot be empty"); }

    for (VertexId v : vertices) {
        if (v >= num_vertices_) { throw std::invalid_argument("Vertex ID out of range"); }
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
    if (edge_id >= hyperedges_.size()) { throw std::invalid_argument("Edge ID out of range"); }
    return hyperedges_[edge_id];
}

const std::vector<Hypergraph::EdgeId>& Hypergraph::get_incident_edges(VertexId vertex_id) const {
    if (vertex_id >= num_vertices_) { throw std::invalid_argument("Vertex ID out of range"); }
    return incident_edges_[vertex_id];
}

void Hypergraph::set_labels(const std::vector<Label>& labels) {
    if (labels.size() != num_vertices_) { throw std::invalid_argument("Labels size must match number of vertices"); }
    labels_ = labels;
}

Hypergraph::FlatHypergraph Hypergraph::flatten() const {
    if (flat_cache_) { return *flat_cache_; } // Return cached version if available

    FlatHypergraph flat_hg;
    flat_hg.num_vertices = get_num_vertices();
    flat_hg.num_edges = get_num_edges();

    // Flatten hyperedges
    flat_hg.edge_offsets.push_back(0);
    for (std::size_t e = 0; e < flat_hg.num_edges; ++e) {
        const auto& vertices = get_hyperedge(e);
        flat_hg.edge_sizes.push_back(vertices.size());

        for (auto v : vertices) { flat_hg.edge_vertices.push_back(v); }
        flat_hg.edge_offsets.push_back(flat_hg.edge_vertices.size());
    }

    // Flatten vertex incident edges
    flat_hg.vertex_offsets.push_back(0);
    for (std::size_t v = 0; v < flat_hg.num_vertices; ++v) {
        const auto& edges = get_incident_edges(v);

        for (auto e : edges) { flat_hg.vertex_edges.push_back(e); }
        flat_hg.vertex_offsets.push_back(flat_hg.vertex_edges.size());
    }

    return flat_hg;
}

void Hypergraph::freeze() {
    // Create and cache the flattened representation
    flat_cache_ = std::make_shared<FlatHypergraph>(flatten());
}

// ---------------------------
// Random generators (common)
// ---------------------------

namespace {

static std::mt19937 make_rng(unsigned int seed) {
    if (seed == 0) {
        std::random_device rd;
        return std::mt19937(rd());
    }
    return std::mt19937(seed);
}

static std::vector<Hypergraph::VertexId> sample_unique_vertices(std::size_t num_vertices, std::size_t k, std::mt19937& gen) {
    if (k > num_vertices) { throw std::invalid_argument("Edge size exceeds number of vertices"); }
    std::set<Hypergraph::VertexId> s;
    std::uniform_int_distribution<std::size_t> vdist(0, num_vertices - 1);
    while (s.size() < k) { s.insert(static_cast<Hypergraph::VertexId>(vdist(gen))); }
    return std::vector<Hypergraph::VertexId>(s.begin(), s.end());
}

static std::vector<Hypergraph::VertexId> sample_unique_from_pool(const std::vector<Hypergraph::VertexId>& pool, std::size_t k, std::mt19937& gen) {
    if (k > pool.size()) { throw std::invalid_argument("Edge size exceeds pool size"); }
    // Reservoir sampling style (simple shuffle then take k)
    std::vector<Hypergraph::VertexId> tmp = pool;
    std::shuffle(tmp.begin(), tmp.end(), gen);
    tmp.resize(k);
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    // Ensure uniqueness; if duplicates removed, refill
    while (tmp.size() < k) {
        // Add random new elements not in tmp
        std::uniform_int_distribution<std::size_t> idx(0, pool.size() - 1);
        Hypergraph::VertexId v = pool[idx(gen)];
        if (!std::binary_search(tmp.begin(), tmp.end(), v)) { tmp.insert(std::upper_bound(tmp.begin(), tmp.end(), v), v); }
    }
    return tmp;
}

} // anonymous namespace

namespace hypergraph_generators {

std::unique_ptr<Hypergraph> generate_uniform(std::size_t num_vertices, std::size_t num_edges, std::size_t min_edge_size, std::size_t max_edge_size, unsigned int seed) {
    if (num_vertices == 0) throw std::invalid_argument("num_vertices must be > 0");
    if (num_edges == 0) throw std::invalid_argument("num_edges must be > 0");
    if (min_edge_size < 2) throw std::invalid_argument("min_edge_size must be >= 2");
    if (max_edge_size < min_edge_size) throw std::invalid_argument("max_edge_size must be >= min_edge_size");

    auto hg = std::make_unique<Hypergraph>(num_vertices);
    auto gen = make_rng(seed);
    std::uniform_int_distribution<std::size_t> sdist(min_edge_size, max_edge_size);

    for (std::size_t e = 0; e < num_edges; ++e) {
        std::size_t k = sdist(gen);
        auto verts = sample_unique_vertices(num_vertices, k, gen);
        hg->add_hyperedge(verts);
    }
    return hg;
}

std::unique_ptr<Hypergraph> generate_fixed_edge_size(std::size_t num_vertices, std::size_t num_edges, std::size_t edge_size, unsigned int seed) {
    if (num_vertices == 0) throw std::invalid_argument("num_vertices must be > 0");
    if (num_edges == 0) throw std::invalid_argument("num_edges must be > 0");
    if (edge_size < 2) throw std::invalid_argument("edge_size must be >= 2");

    auto hg = std::make_unique<Hypergraph>(num_vertices);
    auto gen = make_rng(seed);

    for (std::size_t e = 0; e < num_edges; ++e) {
        auto verts = sample_unique_vertices(num_vertices, edge_size, gen);
        hg->add_hyperedge(verts);
    }
    return hg;
}


std::unique_ptr<Hypergraph>
generate_planted_partition(std::size_t num_vertices, std::size_t num_edges, std::size_t num_communities, double p_intra, std::size_t min_edge_size, std::size_t max_edge_size, unsigned int seed) {
    if (num_vertices == 0) throw std::invalid_argument("num_vertices must be > 0");
    if (num_edges == 0) throw std::invalid_argument("num_edges must be > 0");
    if (num_communities == 0) throw std::invalid_argument("num_communities must be > 0");
    if (min_edge_size < 2) throw std::invalid_argument("min_edge_size must be >= 2");
    if (max_edge_size < min_edge_size) throw std::invalid_argument("max_edge_size must be >= min_edge_size");
    if (p_intra < 0.0 || p_intra > 1.0) throw std::invalid_argument("p_intra must be in [0,1]");

    auto hg = std::make_unique<Hypergraph>(num_vertices);
    auto gen = make_rng(seed);
    std::uniform_int_distribution<std::size_t> sdist(min_edge_size, max_edge_size);
    std::uniform_real_distribution<double> pdist(0.0, 1.0);

    // Partition vertices as evenly as possible
    std::vector<std::vector<Hypergraph::VertexId>> comms(num_communities);
    for (std::size_t v = 0; v < num_vertices; ++v) { comms[v % num_communities].push_back(static_cast<Hypergraph::VertexId>(v)); }

    for (std::size_t e = 0; e < num_edges; ++e) {
        std::size_t k = sdist(gen);
        bool intra = (pdist(gen) < p_intra);
        std::vector<Hypergraph::VertexId> verts;
        verts.reserve(k);

        if (intra) {
            // Choose a community weighted by size
            std::vector<std::size_t> sizes;
            sizes.reserve(num_communities);
            std::size_t total = 0;
            for (const auto& c : comms) {
                sizes.push_back(c.size());
                total += c.size();
            }
            std::uniform_int_distribution<std::size_t> r(0, total - 1);
            std::size_t pick = r(gen);
            std::size_t idx = 0, acc = 0;
            for (; idx < num_communities; ++idx) {
                if (pick < acc + sizes[idx]) break;
                acc += sizes[idx];
            }
            if (idx >= num_communities) idx = num_communities - 1;

            // Sample k unique vertices from the chosen community
            verts = sample_unique_from_pool(comms[idx], std::min(k, comms[idx].size()), gen);
            if (verts.size() < k) {
                // If community smaller than k, add random vertices from outside to fill
                std::vector<Hypergraph::VertexId> pool;
                pool.reserve(num_vertices - comms[idx].size());
                for (std::size_t c = 0; c < num_communities; ++c) {
                    if (c == idx) continue;
                    pool.insert(pool.end(), comms[c].begin(), comms[c].end());
                }
                auto extra = sample_unique_from_pool(pool, k - verts.size(), gen);
                verts.insert(verts.end(), extra.begin(), extra.end());
            }
        } else {
            // Mix: ensure at least two communities appear if possible
            auto all = sample_unique_vertices(num_vertices, k, gen);
            verts = std::move(all);
        }

        hg->add_hyperedge(verts);
    }
    return hg;
}

std::vector<Hypergraph::Label> generate_random_labels(std::size_t num_vertices, std::size_t num_classes, unsigned int seed) {
    if (num_classes == 0) throw std::invalid_argument("num_classes must be > 0");
    auto gen = make_rng(seed);
    std::uniform_int_distribution<Hypergraph::Label> ldist(0, static_cast<int>(num_classes - 1));
    std::vector<Hypergraph::Label> labels(num_vertices);
    for (std::size_t v = 0; v < num_vertices; ++v) { labels[v] = ldist(gen); }
    return labels;
}

std::unique_ptr<Hypergraph>
generate_hsbm(std::size_t num_vertices, std::size_t num_edges, std::size_t num_communities, double p_intra, double p_inter, std::size_t min_edge_size, std::size_t max_edge_size, unsigned int seed) {
    if (num_vertices == 0) throw std::invalid_argument("num_vertices must be > 0");
    if (num_edges == 0) throw std::invalid_argument("num_edges must be > 0");
    if (num_communities == 0) throw std::invalid_argument("num_communities must be > 0");
    if (min_edge_size < 2) throw std::invalid_argument("min_edge_size must be >= 2");
    if (max_edge_size < min_edge_size) throw std::invalid_argument("max_edge_size must be >= min_edge_size");
    if (p_intra < 0.0 || p_intra > 1.0) throw std::invalid_argument("p_intra must be in [0,1]");
    if (p_inter < 0.0 || p_inter > 1.0) throw std::invalid_argument("p_inter must be in [0,1]");

    auto hg = std::make_unique<Hypergraph>(num_vertices);
    auto gen = make_rng(seed);
    std::uniform_int_distribution<std::size_t> sdist(min_edge_size, max_edge_size);
    std::uniform_real_distribution<double> pdist(0.0, 1.0);

    // Partition vertices as evenly as possible (deterministic mapping v % C)
    std::vector<std::vector<Hypergraph::VertexId>> comms(num_communities);
    for (std::size_t v = 0; v < num_vertices; ++v) { comms[v % num_communities].push_back(static_cast<Hypergraph::VertexId>(v)); }

    const std::size_t max_attempts = std::max<std::size_t>(num_edges * 20, 1000);
    std::size_t added = 0;
    std::size_t attempts = 0;
    while (added < num_edges) {
        if (attempts++ > max_attempts) { throw std::runtime_error("hSBM: too many rejections; try increasing p_intra/p_inter or adjusting size range"); }
        const std::size_t k = sdist(gen);
        auto verts = sample_unique_vertices(num_vertices, k, gen);

        // Determine if all vertices are in the same community
        auto comm_of = [&](Hypergraph::VertexId x) { return x % num_communities; };
        const std::size_t base = comm_of(verts[0]);
        bool all_same = true;
        for (std::size_t i = 1; i < verts.size(); ++i) {
            if (comm_of(verts[i]) != base) {
                all_same = false;
                break;
            }
        }

        const double r = pdist(gen);
        const double prob = all_same ? p_intra : p_inter;
        if (r <= prob) {
            hg->add_hyperedge(verts);
            ++added;
        }
    }

    return hg;
}

} // namespace hypergraph_generators

// ---------------------------
// Serialization (binary + JSON)
// ---------------------------

void Hypergraph::save_to_file(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) { throw std::runtime_error("Failed to open file for writing: " + path); }

    const std::uint32_t magic = utils::HGR_ASCII; // 'HGR1'
    const std::uint32_t version = 1u;
    const std::uint64_t nv = static_cast<std::uint64_t>(num_vertices_);
    const std::uint64_t ne = static_cast<std::uint64_t>(get_num_edges());

    os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
    os.write(reinterpret_cast<const char*>(&nv), sizeof(nv));
    os.write(reinterpret_cast<const char*>(&ne), sizeof(ne));

    for (std::size_t e = 0; e < hyperedges_.size(); ++e) {
        const auto& verts = hyperedges_[e];
        const std::uint64_t sz = static_cast<std::uint64_t>(verts.size());
        os.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        for (auto v : verts) {
            const std::uint64_t vv = static_cast<std::uint64_t>(v);
            os.write(reinterpret_cast<const char*>(&vv), sizeof(vv));
        }
    }

    // Always write labels availability flag and labels
    const std::uint8_t has_labels = 1u;
    os.write(reinterpret_cast<const char*>(&has_labels), sizeof(has_labels));
    for (std::size_t v = 0; v < num_vertices_; ++v) {
        const std::int32_t lab = static_cast<std::int32_t>(labels_[v]);
        os.write(reinterpret_cast<const char*>(&lab), sizeof(lab));
    }

    if (!os) { throw std::runtime_error("Failed while writing file: " + path); }
}

std::unique_ptr<Hypergraph> Hypergraph::load_from_file(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { throw std::runtime_error("Failed to open file for reading: " + path); }

    // Detect format by first non-whitespace byte: '{' => JSON, otherwise assume binary
    // We avoid consuming bytes irreversibly by peeking/putback as needed.
    // Since binary starts with 'H' from magic 'HGR1', this is unambiguous.
    // Skip leading whitespace (for JSON)
    while (true) {
        int c = is.peek();
        if (c == EOF) break;
        if (!std::isspace(static_cast<unsigned char>(c))) break;
        is.get();
    }
    int first = is.peek();
    if (first == '{') { return utils::load_hypergraph_from_json_stream(is, path); }
    // Fallback: binary format
    // Reset to beginning before binary read
    is.clear();
    is.seekg(0, std::ios::beg);
    return utils::load_from_binary_stream(is, path);
}
