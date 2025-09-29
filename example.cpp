#include "hypergraph.hpp"
#ifdef BUILD_OPENMP
#include "src/openmp/label_propagation_openmp.hpp"
#endif
#include <iostream>
#include <chrono>
#include <memory>

/**
 * @brief Example usage of hypergraph label propagation algorithms
 */
int main() {
    std::cout << "Hypergraph Label Propagation Example\n";
    std::cout << "=====================================\n\n";
    
    // Create a small example hypergraph
    const std::size_t num_vertices = 6;
    auto hypergraph = std::make_unique<Hypergraph>(num_vertices);
    
    // Add some hyperedges
    // Hyperedge 1: vertices {0, 1, 2}
    hypergraph->add_hyperedge({0, 1, 2});
    
    // Hyperedge 2: vertices {2, 3, 4}  
    hypergraph->add_hyperedge({2, 3, 4});
    
    // Hyperedge 3: vertices {4, 5}
    hypergraph->add_hyperedge({4, 5});
    
    // Hyperedge 4: vertices {0, 3, 5}
    hypergraph->add_hyperedge({0, 3, 5});
    
    // Set initial labels
    std::vector<Hypergraph::Label> initial_labels = {0, 0, 1, 1, 2, 2};
    hypergraph->set_labels(initial_labels);
    
    std::cout << "Initial hypergraph:\n";
    std::cout << "  Vertices: " << hypergraph->get_num_vertices() << "\n";
    std::cout << "  Hyperedges: " << hypergraph->get_num_edges() << "\n";
    
    std::cout << "\nInitial labels: ";
    const auto& labels = hypergraph->get_labels();
    for (std::size_t i = 0; i < labels.size(); ++i) {
        std::cout << "v" << i << "=" << labels[i];
        if (i < labels.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";
    
    std::cout << "Hyperedges:\n";
    for (std::size_t e = 0; e < hypergraph->get_num_edges(); ++e) {
        const auto& vertices = hypergraph->get_hyperedge(e);
        std::cout << "  Edge " << e << ": {";
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            std::cout << vertices[i];
            if (i < vertices.size() - 1) std::cout << ", ";
        }
        std::cout << "}\n";
    }
    std::cout << "\n";
    
#ifdef BUILD_OPENMP
    // Test OpenMP implementation
    std::cout << "Running OpenMP implementation:\n";
    std::cout << "------------------------------\n";
    
    // Make a copy of the hypergraph for OpenMP
    auto openmp_hypergraph = std::make_unique<Hypergraph>(num_vertices);
    for (std::size_t e = 0; e < hypergraph->get_num_edges(); ++e) {
        openmp_hypergraph->add_hyperedge(hypergraph->get_hyperedge(e));
    }
    openmp_hypergraph->set_labels(initial_labels);
    
    LabelPropagationOpenMP openmp_algo(2);  // Use 2 threads
    
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = openmp_algo.run(*openmp_hypergraph, 10);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "  Runtime: " << duration.count() << " μs\n";
    
    std::cout << "  Final labels: ";
    const auto& final_labels = openmp_hypergraph->get_labels();
    for (std::size_t i = 0; i < final_labels.size(); ++i) {
        std::cout << "v" << i << "=" << final_labels[i];
        if (i < final_labels.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";
#else
    std::cout << "OpenMP implementation not available (not compiled with BUILD_OPENMP)\n\n";
#endif
    
    std::cout << "Label propagation analysis:\n";
    std::cout << "---------------------------\n";
    std::cout << "The algorithm propagates labels through hyperedge connectivity.\n";
    std::cout << "Vertices connected by hyperedges tend to adopt similar labels.\n";
    std::cout << "The final labeling represents communities in the hypergraph.\n\n";
    
    return 0;
}