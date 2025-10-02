#include "argparse.hpp"
#include "label_propagation_kokkos.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Hypergraph Label Propagation - Kokkos Implementation\n";
    std::cout << "====================================================\n";

    // Parse command line arguments with cxxopts
    auto opts = CLI::parse_args(argc, argv);
    if (opts.iterations == 0 && opts.help) {
        return 0; // help printed
    }
    CLI::print_cli_summary(opts);

    try {
        std::cout << "Generating test hypergraph...\n";
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

        // Run Kokkos label propagation
        LabelPropagationKokkos algorithm(opts.device);

        auto start_time = std::chrono::high_resolution_clock::now();
        int iterations = algorithm.run(*hypergraph, opts.iterations, opts.tolerance);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\nResults:\n";
        std::cout << "  Iterations: " << iterations << "\n";
        std::cout << "  Runtime: " << duration.count() << " ms\n";
        std::cout << "  Implementation: " << algorithm.get_name() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
