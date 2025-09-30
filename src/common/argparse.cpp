#include "argparse.hpp"
#include "hypergraph.hpp"
#include <cctype>
#include <iostream>
#include <stdexcept>

#include <cxxopts.hpp>

namespace CLI {

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
        ("g,generator", "Generator: uniform|fixed|planted", cxxopts::value<std::string>(out.generator))
        ("uniform", "Shortcut for --generator=uniform", cxxopts::value<bool>(flag_uniform)->default_value("false"))
        ("fixed",   "Shortcut for --generator=fixed",   cxxopts::value<bool>(flag_fixed)->default_value("false"))
        ("planted", "Shortcut for --generator=planted", cxxopts::value<bool>(flag_planted)->default_value("false"))
        ("min-edge-size", "Min edge size (uniform/planted)", cxxopts::value<std::size_t>(out.min_edge_size))
        ("max-edge-size", "Max edge size (uniform/planted)", cxxopts::value<std::size_t>(out.max_edge_size))
        ("edge-size",     "Fixed edge size (fixed)",        cxxopts::value<std::size_t>(out.edge_size))
        ("communities",   "Number of communities (planted)", cxxopts::value<std::size_t>(out.communities))
        ("p-intra",       "Intra-community probability (planted)", cxxopts::value<double>(out.p_intra))
        ("seed",          "Graph RNG seed (0=nondet)", cxxopts::value<unsigned int>(out.seed))
    ;

    // Group: Labels
    opts.add_options("Labels")
        ("label-classes", "Generate random labels in [0,C) (0=skip)", cxxopts::value<std::size_t>(out.label_classes))
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
        std::cout << "             Vertices are chosen uniformly without replacement for each edge.\n";
        std::cout << "             Useful for controlled experiments with constant edge cardinality.\n";
        std::cout << "  planted  : Vertices are partitioned into --communities groups (as evenly as possible).\n";
        std::cout << "             For each edge, size is sampled from [min-edge-size, max-edge-size]. With probability --p-intra,\n";
        std::cout << "             the edge is formed primarily within a single community (filled from outside if needed); otherwise\n";
        std::cout << "             it mixes vertices across communities. Higher --p-intra yields stronger community structure.\n";
        std::cout << "Notes: Use --seed and --label-seed for deterministic generation.\n";
    };

    if (show_version) {
        std::cout << "label_propagation version 0.1\n";
        out.help = true;
        out.iterations = 0;
        return out;
    }

    // Resolve generator from shortcut flags if provided
    int gen_flags = (flag_uniform ? 1 : 0) + (flag_fixed ? 1 : 0) + (flag_planted ? 1 : 0);
    if (gen_flags > 1) {
        std::cerr << "Error: only one of --uniform/--fixed/--planted may be specified.\n";
        print_help_with_generators();
        out.help = true;
        out.iterations = 0;
        return out;
    }
    if (gen_flags == 1) {
        if (flag_uniform) out.generator = "uniform";
        else if (flag_fixed) out.generator = "fixed";
        else out.generator = "planted";
    }

    // Normalize generator to lowercase
    for (auto& c : out.generator) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

    // Validate generator value early for better UX
    if (out.generator != "uniform" && out.generator != "fixed" && out.generator != "planted") {
        std::cerr << "Error: Unknown generator: '" << out.generator << "' (use uniform|fixed|planted).\n";
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
        if (result.count("generator") || flag_uniform || flag_fixed || flag_planted) {
            warn("--load specified: generator selection and parameters are ignored.");
        }
        if (result.count("min-edge-size")) warn("--min-edge-size is ignored when loading from file.");
        if (result.count("max-edge-size")) warn("--max-edge-size is ignored when loading from file.");
        if (result.count("edge-size")) warn("--edge-size is ignored when loading from file.");
        if (result.count("communities")) warn("--communities is ignored when loading from file.");
        if (result.count("p-intra")) warn("--p-intra is ignored when loading from file.");
    } else if (out.generator == "fixed") {
        if (result.count("min-edge-size")) warn("--min-edge-size is ignored with --generator=fixed.");
        if (result.count("max-edge-size")) warn("--max-edge-size is ignored with --generator=fixed.");
        if (!result.count("edge-size")) {
            // Make it explicit to users that edge-size controls this generator
            warn("--edge-size controls fixed generator edge cardinality (default=" + std::to_string(out.edge_size) + ").");
        }
    } else if (out.generator == "uniform") {
        if (result.count("edge-size")) warn("--edge-size is ignored with --generator=uniform.");
        if (result.count("communities")) warn("--communities is ignored with --generator=uniform.");
        if (result.count("p-intra")) warn("--p-intra is ignored with --generator=uniform.");
    } else if (out.generator == "planted") {
        if (result.count("edge-size")) warn("--edge-size is ignored with --generator=planted.");
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

    // Generator-specific validation hints
    if (!out.load_file.empty()) {
        // No further generator validation needed when loading
    } else if (out.generator == "fixed") {
        if (out.edge_size < 2) {
            std::cerr << "Error: --edge-size must be >= 2 for fixed generator.\n";
            print_help_with_generators();
            out.help = true;
            out.iterations = 0;
            return out;
        }
        if (out.edge_size > out.vertices) {
            std::cerr << "Error: --edge-size cannot exceed --vertices.\n";
            print_help_with_generators();
            out.help = true;
            out.iterations = 0;
            return out;
        }
    } else {
        // uniform or planted
        if (out.min_edge_size < 2) {
            std::cerr << "Error: --min-edge-size must be >= 2.\n";
            print_help_with_generators();
            out.help = true;
            out.iterations = 0;
            return out;
        }
        if (out.max_edge_size < out.min_edge_size) {
            std::cerr << "Error: --max-edge-size must be >= --min-edge-size.\n";
            print_help_with_generators();
            out.help = true;
            out.iterations = 0;
            return out;
        }
        if (out.generator == "planted") {
            if (out.communities == 0) {
                std::cerr << "Error: --communities must be > 0 for planted generator.\n";
                print_help_with_generators();
                out.help = true;
                out.iterations = 0;
                return out;
            }
            if (out.communities > out.vertices) {
                std::cerr << "Error: --communities cannot exceed --vertices.\n";
                print_help_with_generators();
                out.help = true;
                out.iterations = 0;
                return out;
            }
            if (out.p_intra < 0.0 || out.p_intra > 1.0) {
                std::cerr << "Error: --p-intra must be within [0,1].\n";
                print_help_with_generators();
                out.help = true;
                out.iterations = 0;
                return out;
            }
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
        } else {
            throw std::invalid_argument("Unknown generator: '" + opts.generator + "' (use uniform|fixed|planted)");
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

} // namespace CLI
