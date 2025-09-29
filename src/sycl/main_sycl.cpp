#include "label_propagation_sycl.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <set>

/**
 * @brief Generate a random hypergraph for testing
 */
std::unique_ptr<Hypergraph> generate_test_hypergraph(std::size_t num_vertices, 
                                                     std::size_t num_edges,
                                                     std::size_t max_edge_size = 5) {
    auto hypergraph = std::make_unique<Hypergraph>(num_vertices);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::size_t> vertex_dist(0, num_vertices - 1);
    std::uniform_int_distribution<std::size_t> size_dist(2, max_edge_size);
    std::uniform_int_distribution<Hypergraph::Label> label_dist(0, num_vertices / 10);

    // Generate random hyperedges
    for (std::size_t e = 0; e < num_edges; ++e) {
        std::size_t edge_size = size_dist(gen);
        std::vector<Hypergraph::VertexId> vertices;
        
        // Generate unique vertices for this edge
        std::set<Hypergraph::VertexId> vertex_set;
        while (vertex_set.size() < edge_size) {
            vertex_set.insert(vertex_dist(gen));
        }
        
        vertices.assign(vertex_set.begin(), vertex_set.end());
        hypergraph->add_hyperedge(vertices);
    }

    // Generate random initial labels
    std::vector<Hypergraph::Label> labels(num_vertices);
    for (std::size_t v = 0; v < num_vertices; ++v) {
        labels[v] = label_dist(gen);
    }
    hypergraph->set_labels(labels);

    return hypergraph;
}

int main(int argc, char* argv[]) {
    std::cout << "Hypergraph Label Propagation - SYCL Implementation\n";
    std::cout << "==================================================\n";

    try {
        // Parse command line arguments
        std::size_t num_vertices = 1000;
        std::size_t num_edges = 5000;
        int max_iterations = 100;
        std::string device_type = "auto";

        if (argc >= 2) num_vertices = std::stoull(argv[1]);
        if (argc >= 3) num_edges = std::stoull(argv[2]);
        if (argc >= 4) max_iterations = std::stoi(argv[3]);
        if (argc >= 5) device_type = argv[4];

        std::cout << "Parameters:\n";
        std::cout << "  Vertices: " << num_vertices << "\n";
        std::cout << "  Edges: " << num_edges << "\n";
        std::cout << "  Max iterations: " << max_iterations << "\n";
        std::cout << "  Device: " << device_type << "\n\n";

        // Generate test hypergraph
        std::cout << "Generating test hypergraph...\n";
        auto hypergraph = generate_test_hypergraph(num_vertices, num_edges);
        
        std::cout << "Hypergraph statistics:\n";
        std::cout << "  Vertices: " << hypergraph->get_num_vertices() << "\n";
        std::cout << "  Hyperedges: " << hypergraph->get_num_edges() << "\n";

        // Create device selector
        std::unique_ptr<sycl::device_selector> selector;
        if (device_type == "gpu") {
            selector = std::make_unique<sycl::gpu_selector>();
        } else if (device_type == "cpu") {
            selector = std::make_unique<sycl::cpu_selector>();
        }

        // Run SYCL label propagation
        LabelPropagationSYCL algorithm(selector.get());
        
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