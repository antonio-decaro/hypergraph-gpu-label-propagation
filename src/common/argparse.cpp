#include "argparse.hpp"
#include "hypergraph.hpp"
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <sstream>

#include <cxxopts.hpp>

namespace CLI {

namespace {
enum class GenKind { Uniform, Fixed, Erdos, Planted, Hsbm, Unknown };

struct FieldDesc { const char* key; const char* label; };
struct GenSpec {
    const char* name;
    std::vector<FieldDesc> fields; // order preserved for printing
};

static const GenSpec& get_spec(GenKind k) {
    static const GenSpec uniform{"uniform", { {"min-edge-size", "min-edge-size"}, {"max-edge-size", "max-edge-size"} }};
    static const GenSpec fixed  {"fixed",   { {"edge-size",     "edge-size"} }};
    static const GenSpec planted{"planted", { {"communities",   "communities"}, {"p-intra", "p-intra"}, {"min-edge-size","min-edge-size"}, {"max-edge-size","max-edge-size"} }};
    static const GenSpec hsbm   {"hsbm",    { {"communities",   "communities"}, {"p-intra","p-intra"}, {"p-inter","p-inter"}, {"min-edge-size","min-edge-size"}, {"max-edge-size","max-edge-size"} }};
    switch (k) {
        case GenKind::Uniform: return uniform;
        case GenKind::Fixed:   return fixed;
        case GenKind::Planted: return planted;
        case GenKind::Hsbm:    return hsbm;
        default:               return uniform; // fallback
    }
}

static GenKind parse_kind(std::string s) {
    for (auto& c : s) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    if (s == "uniform") return GenKind::Uniform;
    if (s == "fixed")   return GenKind::Fixed;
    if (s == "planted") return GenKind::Planted;
    if (s == "hsbm")    return GenKind::Hsbm;
    return GenKind::Unknown;
}

static const std::vector<std::string>& all_gen_keys() {
    static const std::vector<std::string> keys = {
        "min-edge-size", "max-edge-size", "edge-size", "communities", "p-intra", "p-inter"
    };
    return keys;
}

static std::string field_value(const Options& o, const std::string& key) {
    if (key == "min-edge-size") return std::to_string(o.min_edge_size);
    if (key == "max-edge-size") return std::to_string(o.max_edge_size);
    if (key == "edge-size")     return std::to_string(o.edge_size);
    if (key == "communities")   return std::to_string(o.communities);
    if (key == "p-intra")       return std::to_string(o.p_intra);
    if (key == "p-inter")       return std::to_string(o.p_inter);
    return "";
}

static void warn_irrelevant_params(const cxxopts::ParseResult& res, GenKind kind, const std::function<void(const std::string&)>& warn) {
    std::unordered_set<std::string> relevant;
    for (const auto& f : get_spec(kind).fields) relevant.insert(f.key);
    for (const auto& k : all_gen_keys()) {
        if (res.count(k) && !relevant.count(k)) {
            warn(std::string("--") + k + " is ignored with --generator=" + get_spec(kind).name + ".");
        }
    }
}

static bool validate_generator_params(GenKind kind, const Options& o, std::string& err) {
    switch (kind) {
        case GenKind::Fixed:
            if (o.edge_size < 2) { err = "--edge-size must be >= 2 for fixed generator."; return false; }
            if (o.edge_size > o.vertices) { err = "--edge-size cannot exceed --vertices."; return false; }
            return true;
        case GenKind::Uniform:
        case GenKind::Planted:
        case GenKind::Hsbm:
            if (o.min_edge_size < 2) { err = "--min-edge-size must be >= 2."; return false; }
            if (o.max_edge_size < o.min_edge_size) { err = "--max-edge-size must be >= --min-edge-size."; return false; }
            if (kind == GenKind::Planted || kind == GenKind::Hsbm) {
                if (o.communities == 0) { err = "--communities must be > 0."; return false; }
                if (o.communities > o.vertices) { err = "--communities cannot exceed --vertices."; return false; }
                if (o.p_intra < 0.0 || o.p_intra > 1.0) { err = "--p-intra must be within [0,1]."; return false; }
            }
            if (kind == GenKind::Hsbm) {
                if (o.p_inter < 0.0 || o.p_inter > 1.0) { err = "--p-inter must be within [0,1]."; return false; }
            }
            return true;
        default:
            err = "Unknown generator."; return false;
    }
}
} // anonymous namespace

Options parse_args(int argc, char** argv) {
    Options out;
    cxxopts::Options opts("label_propagation", "Hypergraph label propagation (OpenMP / GPU offload)");
    opts.positional_help("");

    // Group: Problem size
    opts.add_options("Problem")
        ("v,vertices", "Number of vertices", cxxopts::value<std::size_t>(out.vertices))
        ("e,edges", "Number of hyperedges", cxxopts::value<std::size_t>(out.edges))
    ;

    // Group: Algorithm
    opts.add_options("Algorithm")
        ("i,iterations", "Max iterations", cxxopts::value<int>(out.iterations))
        ("t,tolerance", "Convergence tolerance", cxxopts::value<double>(out.tolerance))
        ("p,threads", "OpenMP threads (0=auto)", cxxopts::value<int>(out.threads))
    ;

    // Group: Generator selection and parameters
    bool flag_uniform = false, flag_fixed = false, flag_planted = false;
    opts.add_options("Generator")
        ("g,generator", "Generator: uniform|fixed|planted|hsbm", cxxopts::value<std::string>(out.generator))
        ("uniform", "Shortcut for --generator=uniform", cxxopts::value<bool>(flag_uniform)->default_value("false"))
        ("fixed",   "Shortcut for --generator=fixed",   cxxopts::value<bool>(flag_fixed)->default_value("false"))
        ("planted", "Shortcut for --generator=planted", cxxopts::value<bool>(flag_planted)->default_value("false"))
        ("hsbm",  "Shortcut for --generator=hsbm (hypergraph SBM)", cxxopts::value<bool>()->default_value("false"))
        ("min-edge-size", "Min edge size (uniform/planted)", cxxopts::value<std::size_t>(out.min_edge_size))
        ("max-edge-size", "Max edge size (uniform/planted)", cxxopts::value<std::size_t>(out.max_edge_size))
        ("edge-size",     "Fixed edge size (fixed)",        cxxopts::value<std::size_t>(out.edge_size))
        ("communities",   "Number of communities (planted/hsbm)", cxxopts::value<std::size_t>(out.communities))
        ("p-intra",       "Intra-community probability (planted/hsbm)", cxxopts::value<double>(out.p_intra))
        ("p-inter",       "Inter-community probability (hsbm)", cxxopts::value<double>(out.p_inter))
        ("seed",          "Graph RNG seed (0=nondet)", cxxopts::value<unsigned int>(out.seed))
    ;

    // Group: Labels
    opts.add_options("Labels")
        ("label-classes", "Generate random labels in [0,C) (0=skip)", cxxopts::value<std::size_t>(out.label_classes)->default_value("0"))
        ("label-seed",    "Label RNG seed (0=nondet)", cxxopts::value<unsigned int>(out.label_seed))
    ;

    // Group: IO
    opts.add_options("IO")
        ("load", "Load hypergraph from binary file", cxxopts::value<std::string>(out.load_file))
        ("save", "Save hypergraph to binary file",   cxxopts::value<std::string>(out.save_file))
    ;

    // Group: Misc
    bool show_version = false;
    opts.add_options("Misc")
        ("h,help",    "Show help", cxxopts::value<bool>(out.help))
        ("version",    "Show version", cxxopts::value<bool>(show_version)->default_value("false"))
    ;

    opts.parse_positional({});
    auto result = opts.parse(argc, argv);

    // Helper to print help plus generator descriptions
    auto print_help_with_generators = [&]() {
        std::cout << opts.help() << "\n";
        std::cout << "Generators:\n";
        std::cout << "  uniform  : Each hyperedge size is sampled uniformly from [min-edge-size, max-edge-size].\n";
        std::cout << "             Vertices for an edge are chosen uniformly without replacement across all vertices.\n";
        std::cout << "             Produces independent edges with a broad size distribution.\n";
        std::cout << "  fixed    : All hyperedges have exactly --edge-size vertices.\n";
        std::cout << "             Equivalent to d-uniform Erdős-Rényi G^{(d)}(n, m): --edge-size=d, sample m edges uniformly.\n";
        std::cout << "             Vertices are chosen uniformly without replacement for each edge.\n";
        std::cout << "             Useful for controlled experiments with constant edge cardinality.\n";
        std::cout << "  planted  : Vertices are partitioned into --communities groups (as evenly as possible).\n";
        std::cout << "             For each edge, size is sampled from [min-edge-size, max-edge-size]. With probability --p-intra,\n";
        std::cout << "             the edge is formed primarily within a single community (filled from outside if needed); otherwise\n";
        std::cout << "             it mixes vertices across communities. Higher --p-intra yields stronger community structure.\n";
        std::cout << "  hsbm     : Hypergraph SBM via rejection sampling. For each candidate edge, accept with\n";
        std::cout << "             probability --p-intra if all vertices lie in the same community, otherwise with --p-inter.\n";
        std::cout << "             Edge size is drawn from [min-edge-size, max-edge-size].\n";
        std::cout << "Notes: Use --seed and --label-seed for deterministic generation.\n";
    };

    if (show_version) {
        std::cout << "label_propagation version 0.1\n";
        out.help = true;
        out.iterations = 0;
        return out;
    }

    // Resolve generator from shortcut flags if provided
    bool flag_hsbm  = result.count("hsbm")  > 0 ? result["hsbm"].as<bool>()  : false;
    int gen_flags = (flag_uniform ? 1 : 0) + (flag_fixed ? 1 : 0) + (flag_planted ? 1 : 0) + (flag_hsbm ? 1 : 0);
    if (gen_flags > 1) {
        std::cerr << "Error: only one of --uniform/--fixed/--planted/--hsbm may be specified.\n";
        print_help_with_generators();
        out.help = true;
        out.iterations = 0;
        return out;
    }
    if (gen_flags == 1) {
        if (flag_uniform) out.generator = "uniform";
        else if (flag_fixed) out.generator = "fixed";
        else if (flag_planted) out.generator = "planted";
        else out.generator = "hsbm";
    }

    // Normalize generator to lowercase
    for (auto& c : out.generator) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

    // Validate generator value early for better UX
    GenKind kind = parse_kind(out.generator);
    if (kind == GenKind::Unknown) {
        std::cerr << "Error: Unknown generator: '" << out.generator << "' (use uniform|fixed|planted|hsbm).\n";
        print_help_with_generators();
        out.help = true;
        out.iterations = 0;
        return out;
    }

    // Helpful warnings for irrelevant parameters
    auto warned = false;
    auto warn = [&](const std::string& msg) {
        if (!warned) {
            std::cerr << "Note: " << msg << "\n";
        } else {
            std::cerr << msg << "\n";
        }
        warned = true;
    };

    if (!out.load_file.empty()) {
        // Loading overrides generator-specific knobs; warn when both are supplied
        if (result.count("generator") || flag_uniform || flag_fixed || flag_planted || flag_hsbm) {
            warn("--load specified: generator selection and parameters are ignored.");
        }
        for (const auto& k : all_gen_keys()) {
            if (result.count(k)) warn(std::string("--") + k + " is ignored when loading from file.");
        }
    } else {
        warn_irrelevant_params(result, kind, warn);
    }

     // Early parameter sanity for clearer messages than deeper throws
    if (out.vertices == 0) {
        std::cerr << "Error: --vertices must be > 0.\n";
        print_help_with_generators();
        out.help = true;
        out.iterations = 0;
        return out;
    }
    if (out.edges == 0) {
        std::cerr << "Error: --edges must be > 0.\n";
        print_help_with_generators();
        out.help = true;
        out.iterations = 0;
        return out;
    }

    // Generator-specific validation
    if (!out.load_file.empty()) {
        // File dictates structure; skip generator validation
    } else {
        std::string err;
        if (!validate_generator_params(kind, out, err)) {
            std::cerr << "Error: " << err << "\n";
            print_help_with_generators();
            out.help = true;
            out.iterations = 0;
            return out;
        }
    }

    if (out.help) {
        print_help_with_generators();
        // Signal caller to exit by setting iterations to 0
        out.iterations = 0;
        return out;
    }

    return out;
}

std::unique_ptr<Hypergraph> make_hypergraph(const Options& opts) {
    using namespace hypergraph_generators;
    std::unique_ptr<Hypergraph> hg;

    if (!opts.load_file.empty()) {
        // Load from binary file
        hg = Hypergraph::load_from_file(opts.load_file);
    } else {
        if (opts.generator == "uniform") {
            hg = generate_uniform(opts.vertices, opts.edges,
                                  opts.min_edge_size, opts.max_edge_size, opts.seed);
        } else if (opts.generator == "fixed") {
            hg = generate_fixed_edge_size(opts.vertices, opts.edges,
                                          opts.edge_size, opts.seed);
        } else if (opts.generator == "planted") {
            hg = generate_planted_partition(opts.vertices, opts.edges,
                                            opts.communities, opts.p_intra,
                                            opts.min_edge_size, opts.max_edge_size,
                                            opts.seed);
        } else if (opts.generator == "hsbm") {
            hg = generate_hsbm(opts.vertices, opts.edges,
                               opts.communities, opts.p_intra, opts.p_inter,
                               opts.min_edge_size, opts.max_edge_size,
                               opts.seed);
        } else {
            throw std::invalid_argument("Unknown generator: '" + opts.generator + "' (use uniform|fixed|planted|hsbm)");
        }
    }

    if (opts.label_classes > 0) {
        auto labels = generate_random_labels(hg->get_num_vertices(), opts.label_classes, opts.label_seed);
        hg->set_labels(labels);
    }

    if (!opts.save_file.empty()) {
        hg->save_to_file(opts.save_file);
    }

    return hg;
}

void print_cli_summary(const Options& opts) {
    // Minimal, generator-aware line items to avoid verbosity
    std::cout << "Parameters:\n";
    std::cout << "  Max iterations: " << opts.iterations << "\n";
    std::cout << "  Tolerance: " << opts.tolerance << "\n";
    std::cout << "  Threads: " << (opts.threads == 0 ? "auto" : std::to_string(opts.threads)) << "\n";
    if (!opts.load_file.empty()) {
        std::cout << "  Input: " << opts.load_file << "\n";
    } else {
        std::cout << "  Vertices: " << opts.vertices << "\n";
        std::cout << "  Edges: " << opts.edges << "\n";
        std::cout << "  Seed: " << opts.seed << "\n";

        std::cout << "  Generator: " << opts.generator;
        GenKind kind = parse_kind(opts.generator);
        const auto& spec = get_spec(kind);
        if (!spec.fields.empty()) {
            std::cout << " (";
            for (std::size_t i = 0; i < spec.fields.size(); ++i) {
                const auto& f = spec.fields[i];
                std::cout << f.label << "=" << field_value(opts, f.key);
                if (i + 1 < spec.fields.size()) std::cout << ", ";
            }
            std::cout << ")\n";
        } else {
            std::cout << "\n";
        }
    }
    if (opts.label_classes > 0) {
        std::cout << "  Labels: classes=" << opts.label_classes << ", seed=" << opts.label_seed << "\n";
    }
    if (!opts.save_file.empty()) {
        std::cout << "  Output: " << opts.save_file << "\n";
    }
    std::cout << "\n";
}

} // namespace CLI
