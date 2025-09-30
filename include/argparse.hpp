// Simple argument parser using cxxopts (header-only)
// You can provide cxxopts via your system includes or set CXXOPTS_INCLUDE_DIR in CMake.

#pragma once

#include <string>
#include <cstddef>
#include <memory>

class Hypergraph; // forward declaration

namespace CLI {

struct Options {
    // Problem size
    std::size_t vertices = 1000;
    std::size_t edges = 5000;

    // Algorithm
    int iterations = 100;
    double tolerance = 1e-6;
    int threads = 0; // 0 = auto

    // Generator
    std::string generator = "uniform"; // uniform|fixed|planted
    std::size_t min_edge_size = 2;
    std::size_t max_edge_size = 5;
    std::size_t edge_size = 3; // for fixed
    std::size_t communities = 4; // for planted
    double p_intra = 0.8; // for planted
    double p_inter = 0.2; // for hSBM
    unsigned int seed = 0; // 0 = nondeterministic

    // Labels
    std::size_t label_classes = 0; // 0 = leave defaults
    unsigned int label_seed = 0;   // 0 = nondeterministic

    // IO
    std::string load_file;  // if non-empty, load hypergraph from this binary file
    std::string save_file;  // if non-empty, save generated/loaded hypergraph to this binary file

    // Misc
    bool help = false;
};

// Parse CLI arguments (implemented in src/common/argparse.cpp)
Options parse_args(int argc, char** argv);

// Build a hypergraph according to parsed options (implemented in src/common/argparse.cpp)
// - Generates structure using the selected generator and parameters
// - Optionally assigns random labels if label_classes > 0
// Throws std::invalid_argument on invalid generator or parameters
std::unique_ptr<Hypergraph> make_hypergraph(const Options& opts);

} // namespace CLI
