#include "argparse.hpp"
#include "label_propagation_openmp.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Hypergraph Label Propagation - OpenMP Implementation\n";
    std::cout << "===================================================\n";

    // Parse command line arguments with cxxopts
    auto opts = CLI::parse_args(argc, argv);
    if (opts.iterations == 0 && opts.help) {
        return 0; // help printed
    }

    CLI::print_cli_summary(opts);

    // Generate hypergraph via shared argparse helper
    std::unique_ptr<Hypergraph> hypergraph;
    try {
        hypergraph = CLI::make_hypergraph(opts);
        hypergraph->freeze();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }

    std::cout << "Hypergraph statistics:\n";
    std::cout << "  Vertices: " << hypergraph->get_num_vertices() << "\n";
    std::cout << "  Hyperedges: " << hypergraph->get_num_edges() << "\n";
    std::cout << std::endl;

    // Run OpenMP label propagation
    LabelPropagationOpenMP algorithm(opts.device);

    PerformanceMeasurer perf = algorithm.run(*hypergraph, opts.iterations, opts.tolerance);

    std::cout << "\nResults:\n";
    std::cout << "  Iterations: " << perf.iterations() << "\n";
    std::cout << "  Total runtime: " << perf.total_time().count() << " ms\n";
    if (!perf.moments().empty()) {
        std::cout << "  Breakdown:\n";
        for (const auto& moment : perf.moments()) {
            std::cout << "    " << moment.label << ": " << moment.duration.count() << " ms\n";
        }
    }
    std::cout << "  Implementation: " << algorithm.get_name() << "\n";

    return 0;
}
