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

    opts.add_options()
        ("v,vertices", "Number of vertices", cxxopts::value<std::size_t>(out.vertices))
        ("e,edges", "Number of hyperedges", cxxopts::value<std::size_t>(out.edges))
        ("i,iterations", "Max iterations", cxxopts::value<int>(out.iterations))
        ("t,tolerance", "Convergence tolerance", cxxopts::value<double>(out.tolerance))
        ("p,threads", "OpenMP threads (0=auto)", cxxopts::value<int>(out.threads))

        ("g,generator", "Generator: uniform|fixed|planted", cxxopts::value<std::string>(out.generator))
        ("min-edge-size", "Min edge size (uniform/planted)", cxxopts::value<std::size_t>(out.min_edge_size))
        ("max-edge-size", "Max edge size (uniform/planted)", cxxopts::value<std::size_t>(out.max_edge_size))
        ("edge-size", "Fixed edge size (fixed)", cxxopts::value<std::size_t>(out.edge_size))
        ("communities", "Number of communities (planted)", cxxopts::value<std::size_t>(out.communities))
        ("p-intra", "Intra-community probability (planted)", cxxopts::value<double>(out.p_intra))
        ("seed", "Graph RNG seed (0=nondet)", cxxopts::value<unsigned int>(out.seed))

        ("label-classes", "Generate random labels in [0,C) (0=skip)", cxxopts::value<std::size_t>(out.label_classes))
        ("label-seed", "Label RNG seed (0=nondet)", cxxopts::value<unsigned int>(out.label_seed))

        ("h,help", "Show help", cxxopts::value<bool>(out.help))
    ;

    opts.parse_positional({});
    auto result = opts.parse(argc, argv);

    if (out.help) {
        std::cout << opts.help() << "\n";
        // Signal caller to exit by setting iterations to 0
        out.iterations = 0;
        return out;
    }

    // Normalize generator to lowercase
    for (auto& c : out.generator) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

    return out;
}

std::unique_ptr<Hypergraph> make_hypergraph(const Options& opts) {
    using namespace hypergraph_generators;
    std::unique_ptr<Hypergraph> hg;

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

    if (opts.label_classes > 0) {
        auto labels = generate_random_labels(hg->get_num_vertices(), opts.label_classes, opts.label_seed);
        hg->set_labels(labels);
    }

    return hg;
}

} // namespace CLI

