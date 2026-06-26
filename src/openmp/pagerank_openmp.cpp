#include "pagerank_openmp.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

PageRankOpenMP::PageRankOpenMP(const CLI::DeviceOptions& device, double alpha)
    : PageRankAlgorithm(device, alpha) {
    num_threads_ = static_cast<int>(device_.threads);
    if (num_threads_ <= 0) { num_threads_ = omp_get_max_threads(); }
    omp_set_num_threads(num_threads_);
}

PerformanceMeasurer PageRankOpenMP::run(const Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running OpenMP target (GPU) PageRank (alpha=" << alpha_ << ") with " << num_threads_ << " threads\n";

    PerformanceMeasurer perf;
    const auto overall_start = PerformanceMeasurer::clock::now();

    const std::size_t num_vertices = hypergraph.get_num_vertices();
    const std::size_t num_edges    = hypergraph.get_num_edges();

    if (num_vertices == 0 || num_edges == 0) {
        std::cout << "Empty hypergraph detected; nothing to compute.\n";
        perf.set_iterations(0);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
        return perf;
    }

    const auto setup_start = PerformanceMeasurer::clock::now();

    Hypergraph::FlatHypergraph flat = hypergraph.flatten();

    std::vector<float> p(num_vertices, 1.0f / static_cast<float>(num_vertices));
    std::vector<float> h(num_edges, 0.0f);

    const std::size_t nev  = flat.edge_vertices.size();
    const std::size_t neo  = flat.edge_offsets.size();
    const std::size_t nve  = flat.vertex_edges.size();
    const std::size_t nvo  = flat.vertex_offsets.size();

    float* p_ptr             = p.data();
    float* h_ptr             = h.data();
    Hypergraph::VertexId* ev = flat.edge_vertices.data();
    std::size_t* eo          = flat.edge_offsets.data();
    Hypergraph::EdgeId* ve   = flat.vertex_edges.data();
    std::size_t* vo          = flat.vertex_offsets.data();
    std::size_t* es          = flat.edge_sizes.data();

    const float alpha_f = static_cast<float>(alpha_);
    const float inv_n   = 1.0f / static_cast<float>(num_vertices);
    int wgs             = device_.workgroup_size > 0 ? static_cast<int>(device_.workgroup_size) : 256;

    const auto setup_end = PerformanceMeasurer::clock::now();
    perf.add_moment("setup", setup_end - setup_start);

    const auto iteration_start = PerformanceMeasurer::clock::now();
    int iterations_completed = 0;
    bool converged = false;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // Edge phase: h[e] = sum_{u in e} p[u] / d(u)
        {
            const std::size_t num_teams = (num_edges + static_cast<std::size_t>(wgs) - 1) / static_cast<std::size_t>(wgs);
#pragma omp target teams num_teams(num_teams) thread_limit(wgs)                                                                      \
    map(to : p_ptr[0 : num_vertices], ev[0 : nev], eo[0 : neo], vo[0 : nvo]) map(tofrom : h_ptr[0 : num_edges])
#pragma omp distribute parallel for
            for (std::size_t e = 0; e < num_edges; ++e) {
                float acc = 0.0f;
                for (std::size_t i = eo[e]; i < eo[e + 1]; ++i) {
                    const std::size_t u = ev[i];
                    const float du      = static_cast<float>(vo[u + 1] - vo[u]);
                    if (du > 0.0f) acc += p_ptr[u] / du;
                }
                h_ptr[e] = acc;
            }
        }

        // Vertex phase: p[v] = (1-alpha)/|V| + alpha * sum_{e in v} h[e] / delta(e)
        float diff = 0.0f;
        {
            const std::size_t num_teams = (num_vertices + static_cast<std::size_t>(wgs) - 1) / static_cast<std::size_t>(wgs);
#pragma omp target teams num_teams(num_teams) thread_limit(wgs)                                                                        \
    map(to : h_ptr[0 : num_edges], ve[0 : nve], vo[0 : nvo], es[0 : num_edges]) map(tofrom : p_ptr[0 : num_vertices], diff)
#pragma omp distribute parallel for
            for (std::size_t v = 0; v < num_vertices; ++v) {
                float acc = 0.0f;
                for (std::size_t i = vo[v]; i < vo[v + 1]; ++i) {
                    const std::size_t e = ve[i];
                    const float delta_e = static_cast<float>(es[e]);
                    if (delta_e > 0.0f) acc += h_ptr[e] / delta_e;
                }
                const float p_new = (1.0f - alpha_f) * inv_n + alpha_f * acc;
                const float delta  = p_new - p_ptr[v];
                const float absdelta = delta < 0.0f ? -delta : delta;
#pragma omp atomic update
                diff += absdelta;
                p_ptr[v] = p_new;
            }
        }

        if (static_cast<double>(diff) < tolerance) {
            std::cout << "Converged after " << iteration + 1 << " iterations\n";
            converged = true;
            iterations_completed = iteration + 1;
            break;
        }
        if ((iteration + 1) % 10 == 0) { std::cout << "Iteration " << iteration + 1 << " completed\n"; }
    }

    if (!converged) { iterations_completed = max_iterations; }

    const auto iteration_end = PerformanceMeasurer::clock::now();
    perf.add_moment("iterations", iteration_end - iteration_start);

    const auto finalize_start = PerformanceMeasurer::clock::now();
    scores_.assign(p.begin(), p.end());
    const auto finalize_end = PerformanceMeasurer::clock::now();
    perf.add_moment("finalize", finalize_end - finalize_start);

    perf.set_iterations(iterations_completed);
    perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
    return perf;
}
