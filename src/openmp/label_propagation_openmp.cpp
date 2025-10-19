#include "label_propagation_openmp.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>

namespace {
struct ExecutionPool {
    std::vector<std::uint32_t> wg_pool_edges;
    std::vector<std::uint32_t> sg_pool_edges;
    std::vector<std::uint32_t> wi_pool_edges;
    std::vector<std::uint32_t> wg_pool_vertices;
    std::vector<std::uint32_t> sg_pool_vertices;
    std::vector<std::uint32_t> wi_pool_vertices;
};

ExecutionPool create_execution_pool(const Hypergraph& hypergraph) {
    ExecutionPool pool{};
    const std::size_t num_edges = hypergraph.get_num_edges();
    const std::size_t num_vertices = hypergraph.get_num_vertices();

    for (std::size_t e = 0; e < num_edges; ++e) {
        const auto edge_size = hypergraph.get_hyperedge(e).size();
        if (edge_size > 256) {
            pool.wg_pool_edges.push_back(static_cast<std::uint32_t>(e));
        } else if (edge_size > 32) {
            pool.sg_pool_edges.push_back(static_cast<std::uint32_t>(e));
        } else {
            pool.wi_pool_edges.push_back(static_cast<std::uint32_t>(e));
        }
    }

    for (std::size_t v = 0; v < num_vertices; ++v) {
        const auto incident_size = hypergraph.get_incident_edges(v).size();
        if (incident_size > 1024) {
            pool.wg_pool_vertices.push_back(static_cast<std::uint32_t>(v));
        } else if (incident_size > 256) {
            pool.sg_pool_vertices.push_back(static_cast<std::uint32_t>(v));
        } else {
            pool.wi_pool_vertices.push_back(static_cast<std::uint32_t>(v));
        }
    }

    return pool;
}
} // namespace

LabelPropagationOpenMP::LabelPropagationOpenMP(const CLI::DeviceOptions& device) : LabelPropagationAlgorithm(device) {
    num_threads_ = static_cast<int>(device_.threads);
    if (num_threads_ <= 0) { num_threads_ = omp_get_max_threads(); }
    omp_set_num_threads(num_threads_);
}

PerformanceMeasurer LabelPropagationOpenMP::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running OpenMP target (GPU) label propagation with " << num_threads_ << " threads\n";

    PerformanceMeasurer perf;
    const auto overall_start = PerformanceMeasurer::clock::now();

    const std::size_t num_vertices = hypergraph.get_num_vertices();
    const std::size_t num_edges = hypergraph.get_num_edges();

    if (num_vertices == 0 || num_edges == 0) {
        std::cout << "Empty hypergraph detected; nothing to compute.\n";
        perf.set_iterations(0);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
        return perf;
    }

    const auto setup_start = PerformanceMeasurer::clock::now();
    // Flatten hypergraph for device-friendly access
    Hypergraph::FlatHypergraph flat = hypergraph.flatten();

    // Host-side labels: vertex labels initialized from input; edge labels start at 0
    std::vector<Hypergraph::Label> vertex_labels = hypergraph.get_labels();
    std::vector<Hypergraph::Label> edge_labels(num_edges, 0);

    // Cache sizes used in OpenMP map clauses
    const std::size_t edge_vertices_size = flat.edge_vertices.size();
    const std::size_t edge_offsets_size = flat.edge_offsets.size();
    const std::size_t vertex_edges_size = flat.vertex_edges.size();
    const std::size_t vertex_offsets_size = flat.vertex_offsets.size();

    // SYCL version assumes small bounded label space; mirror that here
    const int max_labels = static_cast<int>(device_.max_labels);
    if (max_labels <= 0) { throw std::invalid_argument("device.max_labels must be > 0"); }
    constexpr int MAX_LABELS_CAP = 10;

    const auto setup_end = PerformanceMeasurer::clock::now();
    perf.add_moment("setup", setup_end - setup_start);

    if (max_labels > MAX_LABELS_CAP) { throw std::invalid_argument("device.max_labels must be <= MAX_LABELS_CAP"); }

    auto exec_pool = create_execution_pool(hypergraph);

    const auto iteration_start = PerformanceMeasurer::clock::now();
    int iterations_completed = 0;
    bool converged = false;
    constexpr int MAX_TEAM_SIZE = 1024;
    int wgs = device_.workgroup_size > 0 ? static_cast<int>(device_.workgroup_size) : 256;
    wgs = std::min(wgs, MAX_TEAM_SIZE);
    const int sg_threads = std::max(1, std::min(wgs, 32));
    constexpr int wi_threads = 1;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Hypergraph::Label* vlabels = vertex_labels.data();
        Hypergraph::Label* elabels = edge_labels.data();
        const Hypergraph::VertexId* edge_vertices = flat.edge_vertices.data();
        const std::size_t* edge_offsets = flat.edge_offsets.data();
        const Hypergraph::EdgeId* vertex_edges = flat.vertex_edges.data();
        const std::size_t* vertex_offsets = flat.vertex_offsets.data();

        auto run_edge_pool = [&](const std::vector<std::uint32_t>& pool_edges, int team_size) {
            if (pool_edges.empty()) { return; }
            team_size = std::max(1, team_size);
            const std::size_t pool_size = pool_edges.size();
            const std::uint32_t* pool_ptr = pool_edges.data();
#pragma omp target teams num_teams(pool_size) thread_limit(team_size) map(to : vlabels[0 : num_vertices], edge_vertices[0 : edge_vertices_size], edge_offsets[0 : edge_offsets_size], pool_ptr[0 : pool_size]) map(tofrom : elabels[0 : num_edges])
            {
                const std::size_t team_id = static_cast<std::size_t>(omp_get_team_num());
                if (team_id < pool_size) {
                    const std::size_t e = static_cast<std::size_t>(pool_ptr[team_id]);
                    const std::size_t v_begin = edge_offsets[e];
                    const std::size_t v_end = edge_offsets[e + 1];
                    const std::size_t degree = v_end - v_begin;
                    float counts[MAX_LABELS_CAP];
#pragma omp parallel shared(counts)
                    {
                        const int tid = omp_get_thread_num();
                        const int team_threads = omp_get_num_threads();
                        for (int lab = tid; lab < max_labels; lab += team_threads) { counts[lab] = 0.0f; }
#pragma omp barrier
                        for (std::size_t idx = tid; idx < degree; idx += static_cast<std::size_t>(team_threads)) {
                            const auto vertex = edge_vertices[v_begin + idx];
                            const int lab = static_cast<int>(vlabels[vertex]);
                            if (lab >= 0 && lab < max_labels) {
#pragma omp atomic update
                                counts[lab] += 1.0f;
                            }
                        }
#pragma omp barrier
                        if (tid == 0) {
                            Hypergraph::Label best = elabels[e];
                            float best_w = -1.0f;
                            for (int lab = 0; lab < max_labels; ++lab) {
                                const float w = counts[lab];
                                if (w > best_w) {
                                    best_w = w;
                                    best = static_cast<Hypergraph::Label>(lab);
                                }
                            }
                            elabels[e] = best;
                        }
                    }
                }
            }
        };

        run_edge_pool(exec_pool.wg_pool_edges, wgs);
        run_edge_pool(exec_pool.sg_pool_edges, sg_threads);
        run_edge_pool(exec_pool.wi_pool_edges, wi_threads);

        std::size_t changes = 0;
        auto run_vertex_pool = [&](const std::vector<std::uint32_t>& pool_vertices, int team_size) {
            if (pool_vertices.empty()) { return; }
            team_size = std::max(1, team_size);
            const std::size_t pool_size = pool_vertices.size();
            const std::uint32_t* pool_ptr = pool_vertices.data();
#pragma omp target teams num_teams(pool_size) thread_limit(team_size) map(to : elabels[0 : num_edges], vertex_edges[0 : vertex_edges_size], vertex_offsets[0 : vertex_offsets_size], pool_ptr[0 : pool_size]) map(tofrom : vlabels[0 : num_vertices], changes)
            {
                const std::size_t team_id = static_cast<std::size_t>(omp_get_team_num());
                if (team_id < pool_size) {
                    const std::size_t v = static_cast<std::size_t>(pool_ptr[team_id]);
                    const std::size_t e_begin = vertex_offsets[v];
                    const std::size_t e_end = vertex_offsets[v + 1];
                    const std::size_t degree = e_end - e_begin;
                    float counts[MAX_LABELS_CAP];
                    std::size_t team_changes = 0;
#pragma omp parallel shared(counts, team_changes)
                    {
                        const int tid = omp_get_thread_num();
                        const int team_threads = omp_get_num_threads();
                        for (int lab = tid; lab < max_labels; lab += team_threads) { counts[lab] = 0.0f; }
#pragma omp barrier
                        for (std::size_t idx = tid; idx < degree; idx += static_cast<std::size_t>(team_threads)) {
                            const auto edge = vertex_edges[e_begin + idx];
                            const int lab = static_cast<int>(elabels[edge]);
                            if (lab >= 0 && lab < max_labels) {
#pragma omp atomic update
                                counts[lab] += 1.0f;
                            }
                        }
#pragma omp barrier
                        if (tid == 0) {
                            Hypergraph::Label current = vlabels[v];
                            Hypergraph::Label best = current;
                            float best_w = -1.0f;
                            for (int lab = 0; lab < max_labels; ++lab) {
                                const float w = counts[lab];
                                if (w > best_w) {
                                    best_w = w;
                                    best = static_cast<Hypergraph::Label>(lab);
                                }
                            }
                            if (best != current) {
                                vlabels[v] = best;
#pragma omp atomic update
                                team_changes += 1;
                            }
                        }
                    }
#pragma omp atomic update
                    changes += team_changes;
                }
            }
        };

        run_vertex_pool(exec_pool.wg_pool_vertices, wgs);
        run_vertex_pool(exec_pool.sg_pool_vertices, sg_threads);
        run_vertex_pool(exec_pool.wi_pool_vertices, wi_threads);

        const double change_ratio = static_cast<double>(changes) / static_cast<double>(num_vertices);
        if (change_ratio < tolerance) {
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
    hypergraph.set_labels(vertex_labels);
    const auto finalize_end = PerformanceMeasurer::clock::now();
    perf.add_moment("finalize", finalize_end - finalize_start);

    perf.set_iterations(iterations_completed);
    perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
    return perf;
}
