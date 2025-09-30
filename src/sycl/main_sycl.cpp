#include "label_propagation_sycl.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <set>


int main(int argc, char* argv[]) {
    std::cout << "Hypergraph Label Propagation - SYCL Implementation\n";
    std::cout << "==================================================\n";

    try {
        // Parse command line arguments
        std::size_t num_vertices = 1000;
        std::size_t num_edges = 5000;
        int max_iterations = 100;

        if (argc >= 2) num_vertices = std::stoull(argv[1]);
        if (argc >= 3) num_edges = std::stoull(argv[2]);
        if (argc >= 4) max_iterations = std::stoi(argv[3]);

        std::cout << "Parameters:\n";
        std::cout << "  Vertices: " << num_vertices << "\n";
        std::cout << "  Edges: " << num_edges << "\n";
        std::cout << "  Max iterations: " << max_iterations << "\n";

        // Generate test hypergraph
        std::cout << "Generating test hypergraph...\n";
        auto hypergraph = generate_test_hypergraph(num_vertices, num_edges);
        
        std::cout << "Hypergraph statistics:\n";
        std::cout << "  Vertices: " << hypergraph->get_num_vertices() << "\n";
        std::cout << "  Hyperedges: " << hypergraph->get_num_edges() << "\n";

        sycl::queue queue {sycl::gpu_selector_v};

        // Run SYCL label propagation
        LabelPropagationSYCL algorithm(queue);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        int iterations = algorithm.run(*hypergraph, max_iterations);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\nResults:\n";
        std::cout << "  Iterations: " << iterations << "\n";
        std::cout << "  Runtime: " << duration.count() << " ms\n";
        std::cout << "  Implementation: " << algorithm.get_name() << "\n";

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}