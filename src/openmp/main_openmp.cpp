#include "label_propagation_openmp.hpp"
#include "argparse.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "Hypergraph Label Propagation - OpenMP Implementation\n";
    std::cout << "===================================================\n";

    // Parse command line arguments with cxxopts
    auto opts = CLI::parse_args(argc, argv);
    if (opts.iterations == 0 && opts.help) {
        return 0; // help printed
    }

    std::cout << "Parameters:\n";
    std::cout << "  Vertices: " << opts.vertices << "\n";
    std::cout << "  Edges: " << opts.edges << "\n";
    std::cout << "  Max iterations: " << opts.iterations << "\n";
    std::cout << "  Tolerance: " << opts.tolerance << "\n";
    std::cout << "  Threads: " << (opts.threads == 0 ? "auto" : std::to_string(opts.threads)) << "\n";
    std::cout << "  Generator: " << opts.generator << "\n";
    std::cout << "  Seed: " << opts.seed << "\n\n";

    // Generate hypergraph via shared argparse helper
    std::unique_ptr<Hypergraph> hypergraph;
    try {
        hypergraph = CLI::make_hypergraph(opts);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }

    std::cout << "Hypergraph statistics:\n";
    std::cout << "  Vertices: " << hypergraph->get_num_vertices() << "\n";
    std::cout << "  Hyperedges: " << hypergraph->get_num_edges() << "\n";

    // Run OpenMP label propagation
    LabelPropagationOpenMP algorithm(opts.threads);

    auto start_time = std::chrono::high_resolution_clock::now();
    int iterations = algorithm.run(*hypergraph, opts.iterations, opts.tolerance);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nResults:\n";
    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "  Runtime: " << duration.count() << " ms\n";
    std::cout << "  Implementation: " << algorithm.get_name() << "\n";

    return 0;
}
