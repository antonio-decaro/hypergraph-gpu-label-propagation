#include "argparse.hpp"
#include "label_propagation_cuda.hpp"
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Hypergraph Label Propagation - CUDA Implementation\n";
    std::cout << "=================================================\n";

    auto opts = CLI::parse_args(argc, argv);
    if (opts.iterations == 0 && opts.help) { return 0; }
    CLI::print_cli_summary(opts);

    try {
        std::cout << "Generating hypergraph...\n";
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

        LabelPropagationCUDA algorithm(opts.device);

        const auto start_time = std::chrono::high_resolution_clock::now();
        const int iterations = algorithm.run(*hypergraph, opts.iterations, opts.tolerance);
        const auto end_time = std::chrono::high_resolution_clock::now();

        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\nResults:\n";
        std::cout << "  Iterations: " << iterations << "\n";
        std::cout << "  Runtime: " << duration.count() << " ms\n";
        std::cout << "  Implementation: " << algorithm.get_name() << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "CUDA error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
