#pragma once

#include "hypergraph.hpp"

#include <cctype>
#include <istream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utils {

constexpr std::uint32_t HGR_ASCII = 0x31475248; // 'H''G''R''1'

// Minimal JSON reader for the supported hypergraph schema.
// Supported schemas documented in utils.hpp.
struct JsonIn {
    std::istream& is;
    explicit JsonIn(std::istream& in) : is(in) {}
    void skip_ws() {
        while (true) {
            int c = is.peek();
            if (c == EOF) return;
            if (!std::isspace(static_cast<unsigned char>(c))) return;
            is.get();
        }
    }
    void expect(char ch) {
        skip_ws();
        int c = is.get();
        if (c != ch) throw std::runtime_error(std::string("JSON parse error: expected '") + ch + "'");
    }
    bool try_consume(char ch) {
        skip_ws();
        int c = is.peek();
        if (c == ch) {
            is.get();
            return true;
        }
        return false;
    }
    std::string parse_string() {
        skip_ws();
        if (is.get() != '"') throw std::runtime_error("JSON parse error: expected string");
        std::string out;
        while (true) {
            int c = is.get();
            if (c == EOF) throw std::runtime_error("JSON parse error: unterminated string");
            if (c == '"') break;
            if (c == '\\') {
                int e = is.get();
                if (e == EOF) throw std::runtime_error("JSON parse error: bad escape");
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: throw std::runtime_error("JSON parse error: unsupported escape");
                }
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
        return out;
    }
    // Parses a non-negative integer (number token). No exponent/float.
    std::uint64_t parse_uint() {
        skip_ws();
        std::string buf;
        int c = is.peek();
        if (!std::isdigit(static_cast<unsigned char>(c))) throw std::runtime_error("JSON parse error: expected unsigned integer");
        while (true) {
            int d = is.peek();
            if (d == EOF || !std::isdigit(static_cast<unsigned char>(d))) break;
            buf.push_back(static_cast<char>(is.get()));
        }
        if (buf.empty()) throw std::runtime_error("JSON parse error: empty unsigned integer");
        std::uint64_t v = 0;
        for (char ch : buf) v = v * 10u + static_cast<unsigned>(ch - '0');
        return v;
    }
};

// Utility to skip arbitrary JSON value (object/array/string/number/literal)
static void skip_value(JsonIn& j) {
    j.skip_ws();
    int c = j.is.peek();
    if (c == '"') {
        (void)j.parse_string();
    } else if (c == '{') {
        int depth = 0;
        do {
            int ch = j.is.get();
            if (ch == '{')
                depth++;
            else if (ch == '}')
                depth--;
            else if (ch == '"') {
                do {
                    int sc = j.is.get();
                    if (sc == '\\')
                        (void)j.is.get();
                    else if (sc == '"')
                        break;
                    else if (sc == EOF)
                        throw std::runtime_error("JSON parse error: unterminated string while skipping");
                } while (true);
            }
            if (ch == EOF) throw std::runtime_error("JSON parse error: unterminated object while skipping");
        } while (depth > 0);
    } else if (c == '[') {
        int depth = 0;
        do {
            int ch = j.is.get();
            if (ch == '[')
                depth++;
            else if (ch == ']')
                depth--;
            else if (ch == '"') {
                do {
                    int sc = j.is.get();
                    if (sc == '\\')
                        (void)j.is.get();
                    else if (sc == '"')
                        break;
                    else if (sc == EOF)
                        throw std::runtime_error("JSON parse error: unterminated string while skipping");
                } while (true);
            }
            if (ch == EOF) throw std::runtime_error("JSON parse error: unterminated array while skipping");
        } while (depth > 0);
    } else {
        // number, true, false, null: consume token chars
        while (true) {
            int ch = j.is.peek();
            if (ch == EOF) break;
            if (std::isspace(static_cast<unsigned char>(ch)) || ch == ',' || ch == '}' || ch == ']') break;
            j.is.get();
        }
    }
}

} // namespace utils

namespace utils {

std::unique_ptr<Hypergraph> load_hypergraph_from_json_stream(std::istream& in, const std::string& source_hint) {
    (void)source_hint; // currently only used for error context in throws if desired

    JsonIn j(in);
    j.skip_ws();
    if (!j.try_consume('{')) throw std::runtime_error("JSON parse error: expected '{'");

    std::size_t num_vertices = 0;
    std::vector<std::vector<Hypergraph::VertexId>> edges;
    std::vector<Hypergraph::Label> labels;

    bool saw_type_hg = false;
    bool saw_node_data = false;
    bool saw_edge_dict = false;
    std::unordered_map<std::string, Hypergraph::VertexId> idmap;
    std::vector<std::vector<Hypergraph::VertexId>> edges_alt;
    auto ensure_id = [&](const std::string& sid) -> Hypergraph::VertexId {
        auto it = idmap.find(sid);
        if (it != idmap.end()) return it->second;
        Hypergraph::VertexId nid = static_cast<Hypergraph::VertexId>(idmap.size());
        idmap.emplace(sid, nid);
        return nid;
    };

    bool first = true;
    while (true) {
        j.skip_ws();
        if (j.try_consume('}')) break;
        if (!first) j.expect(',');
        j.skip_ws();
        std::string key = j.parse_string();
        j.expect(':');
        if (key == "num_vertices" || key == "vertices" || key == "numVertices") {
            std::uint64_t v = j.parse_uint();
            if (v == 0) throw std::runtime_error("JSON: num_vertices must be > 0");
            num_vertices = static_cast<std::size_t>(v);
        } else if (key == "edges" || key == "hyperedges") {
            j.skip_ws();
            j.expect('[');
            bool inner_first = true;
            while (true) {
                j.skip_ws();
                if (j.try_consume(']')) break; // no more edges
                if (!inner_first) j.expect(',');
                j.skip_ws();
                j.expect('[');
                std::vector<Hypergraph::VertexId> e;
                bool e_first = true;
                while (true) {
                    j.skip_ws();
                    if (j.try_consume(']')) break; // end one edge
                    if (!e_first) j.expect(',');
                    std::uint64_t vv = j.parse_uint();
                    e.push_back(static_cast<Hypergraph::VertexId>(vv));
                    e_first = false;
                }
                if (e.empty()) throw std::runtime_error("JSON: hyperedge cannot be empty");
                edges.push_back(std::move(e));
                inner_first = false;
            }
        } else if (key == "labels") {
            j.skip_ws();
            j.expect('[');
            bool l_first = true;
            while (true) {
                j.skip_ws();
                if (j.try_consume(']')) break;
                if (!l_first) j.expect(',');
                std::uint64_t lv = j.parse_uint();
                labels.push_back(static_cast<Hypergraph::Label>(lv));
                l_first = false;
            }
        } else if (key == "type") {
            std::string v = j.parse_string();
            if (v == "hypergraph") saw_type_hg = true;
        } else if (key == "hypergraph-data") {
            skip_value(j); // metadata, ignored
        } else if (key == "node-data") {
            j.skip_ws();
            j.expect('{');
            bool nd_first = true;
            while (true) {
                j.skip_ws();
                if (j.try_consume('}')) break;
                if (!nd_first) j.expect(',');
                std::string nid = j.parse_string();
                ensure_id(nid);
                j.expect(':');
                skip_value(j); // skip node attributes
                nd_first = false;
            }
            saw_node_data = true;
        } else if (key == "edge-dict") {
            j.skip_ws();
            j.expect('{');
            bool ed_first = true;
            while (true) {
                j.skip_ws();
                if (j.try_consume('}')) break;
                if (!ed_first) j.expect(',');
                std::string eid = j.parse_string();
                (void)eid; // not used
                j.expect(':');
                j.skip_ws();
                j.expect('[');
                std::vector<Hypergraph::VertexId> evec;
                bool arr_first = true;
                while (true) {
                    j.skip_ws();
                    if (j.try_consume(']')) break;
                    if (!arr_first) j.expect(',');
                    std::string nid = j.parse_string();
                    evec.push_back(ensure_id(nid));
                    arr_first = false;
                }
                if (evec.empty()) throw std::runtime_error("JSON: hyperedge cannot be empty");
                edges_alt.push_back(std::move(evec));
                ed_first = false;
            }
            saw_edge_dict = true;
        } else {
            skip_value(j);
        }
        first = false;
    }

    if (saw_type_hg || saw_node_data || saw_edge_dict) {
        const std::size_t nv = idmap.size();
        if (nv == 0) throw std::runtime_error("JSON: no vertices found in node-data/edge-dict");
        auto hg = std::make_unique<Hypergraph>(nv);
        for (const auto& e : edges_alt) hg->add_hyperedge(e);
        if (!labels.empty()) {
            if (labels.size() != nv) throw std::runtime_error("JSON: labels size must equal number of vertices");
            hg->set_labels(labels);
        }
        return hg;
    } else {
        if (num_vertices == 0) throw std::runtime_error("JSON: missing/invalid num_vertices");
        auto hg = std::make_unique<Hypergraph>(num_vertices);
        for (const auto& e : edges) hg->add_hyperedge(e);
        if (!labels.empty()) {
            if (labels.size() != num_vertices) throw std::runtime_error("JSON: labels size must equal num_vertices");
            hg->set_labels(labels);
        }
        return hg;
    }
}

static std::unique_ptr<Hypergraph> load_from_binary_stream(std::istream& is, const std::string& path) {
    std::uint32_t magic = 0, version = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!is || magic != HGR_ASCII || version != 1u) { throw std::runtime_error("Invalid hypergraph file (bad magic/version): " + path); }

    std::uint64_t nv = 0, ne = 0;
    is.read(reinterpret_cast<char*>(&nv), sizeof(nv));
    is.read(reinterpret_cast<char*>(&ne), sizeof(ne));
    if (!is || nv == 0) { throw std::runtime_error("Invalid hypergraph file (bad header): " + path); }

    auto hg = std::make_unique<Hypergraph>(static_cast<std::size_t>(nv));

    for (std::uint64_t e = 0; e < ne; ++e) {
        std::uint64_t sz = 0;
        is.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        if (!is || sz == 0) { throw std::runtime_error("Invalid hypergraph file (bad edge size): " + path); }
        std::vector<Hypergraph::VertexId> verts;
        verts.reserve(static_cast<std::size_t>(sz));
        for (std::uint64_t i = 0; i < sz; ++i) {
            std::uint64_t vv = 0;
            is.read(reinterpret_cast<char*>(&vv), sizeof(vv));
            if (!is) { throw std::runtime_error("Invalid hypergraph file (truncated vertices): " + path); }
            verts.push_back(static_cast<Hypergraph::VertexId>(vv));
        }
        hg->add_hyperedge(verts);
    }

    // Attempt to read labels flag (optional for forward compatibility)
    std::uint8_t has_labels = 0u;
    is.read(reinterpret_cast<char*>(&has_labels), sizeof(has_labels));
    if (is && has_labels) {
        std::vector<Hypergraph::Label> labels(static_cast<std::size_t>(nv), 0);
        for (std::size_t v = 0; v < static_cast<std::size_t>(nv); ++v) {
            std::int32_t lab = 0;
            is.read(reinterpret_cast<char*>(&lab), sizeof(lab));
            if (!is) { throw std::runtime_error("Invalid hypergraph file (truncated labels): " + path); }
            labels[v] = static_cast<Hypergraph::Label>(lab);
        }
        hg->set_labels(labels);
    }

    return hg;
}

} // namespace utils
