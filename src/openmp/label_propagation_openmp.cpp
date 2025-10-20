#include "label_propagation_openmp.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct ExecutionPool {
    std::vector<std::uint32_t> wg_pool_edges;
    std::vector<std::uint32_t> wi_pool_edges;
    std::vector<std::uint32_t> wg_pool_vertices;
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
        } else {
            pool.wi_pool_edges.push_back(static_cast<std::uint32_t>(e));
        }
    }

    for (std::size_t v = 0; v < num_vertices; ++v) {
        const auto incident_size = hypergraph.get_incident_edges(v).size();
        if (incident_size > 1024) {
            pool.wg_pool_vertices.push_back(static_cast<std::uint32_t>(v));
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
    Hypergraph::Label* vlabels = vertex_labels.data();
    Hypergraph::Label* elabels = edge_labels.data();
    const Hypergraph::VertexId* edge_vertices = flat.edge_vertices.data();
    const std::size_t* edge_offsets = flat.edge_offsets.data();
    const Hypergraph::EdgeId* vertex_edges = flat.vertex_edges.data();
    const std::size_t* vertex_offsets = flat.vertex_offsets.data();
#pragma omp target data map(to : edge_vertices[0 : edge_vertices_size], edge_offsets[0 : edge_offsets_size], vertex_edges[0 : vertex_edges_size], vertex_offsets[0 : vertex_offsets_size]) map(tofrom : vlabels[0 : num_vertices], elabels[0 : num_edges])
    {
        auto run_edge_pool_wg = [&](const std::vector<std::uint32_t>& pool_edges, int team_size) {
            if (pool_edges.empty()) { return; }
            team_size = std::max(1, team_size);
            const std::size_t pool_size = pool_edges.size();
            const std::uint32_t* pool_ptr = pool_edges.data();
#pragma omp target teams num_teams(pool_size) thread_limit(team_size) map(present, to : edge_vertices[0 : edge_vertices_size], edge_offsets[0 : edge_offsets_size]) map(present, tofrom : vlabels[0 : num_vertices], elabels[0 : num_edges]) map(to : pool_ptr[0 : pool_size])
            {
                const std::size_t team_id = static_cast<std::size_t>(omp_get_team_num());
                if (team_id < pool_size) {
                    const std::size_t e = static_cast<std::size_t>(pool_ptr[team_id]);
                    const std::size_t v_begin = edge_offsets[e];
                    const std::size_t v_end = edge_offsets[e + 1];
                    const std::size_t degree = v_end - v_begin;
                    float counts[MAX_LABELS_CAP] = {0.0f};
#pragma omp parallel for reduction(+ : counts[:MAX_LABELS_CAP])
                    for (std::size_t idx = 0; idx < degree; ++idx) {
                        const auto vertex = edge_vertices[v_begin + idx];
                        const int lab = static_cast<int>(vlabels[vertex]);
                        if (lab >= 0 && lab < max_labels) { counts[lab] += 1.0f; }
                    }
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
        };

        auto run_edge_pool_wi = [&](const std::vector<std::uint32_t>& pool_edges) {
            if (pool_edges.empty()) { return; }
            const std::size_t pool_size = pool_edges.size();
            const std::uint32_t* pool_ptr = pool_edges.data();
#pragma omp target teams distribute parallel for map(present, to : edge_vertices[0 : edge_vertices_size], edge_offsets[0 : edge_offsets_size]) map(present, tofrom : vlabels[0 : num_vertices], elabels[0 : num_edges]) map(to : pool_ptr[0 : pool_size])
            for (std::size_t idx = 0; idx < pool_size; ++idx) {
                const std::size_t e = static_cast<std::size_t>(pool_ptr[idx]);
                const std::size_t v_begin = edge_offsets[e];
                const std::size_t v_end = edge_offsets[e + 1];
                float counts[MAX_LABELS_CAP] = {0.0f};
                for (std::size_t v_idx = v_begin; v_idx < v_end; ++v_idx) {
                    const auto vertex = edge_vertices[v_idx];
                    const int lab = static_cast<int>(vlabels[vertex]);
                    if (lab >= 0 && lab < max_labels) { counts[lab] += 1.0f; }
                }
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
        };

        auto run_vertex_pool_wg = [&](const std::vector<std::uint32_t>& pool_vertices, int team_size) -> std::size_t {
            if (pool_vertices.empty()) { return 0; }
            team_size = std::max(1, team_size);
            const std::size_t pool_size = pool_vertices.size();
            const std::uint32_t* pool_ptr = pool_vertices.data();
            std::vector<std::uint32_t> team_changes(pool_size, 0);
            std::uint32_t* team_changes_ptr = team_changes.data();
#pragma omp target teams num_teams(pool_size) thread_limit(team_size) map(present, to : vertex_edges[0 : vertex_edges_size], vertex_offsets[0 : vertex_offsets_size]) map(present, tofrom : vlabels[0 : num_vertices], elabels[0 : num_edges]) map(to : pool_ptr[0 : pool_size]) map(tofrom : team_changes_ptr[0 : pool_size])
            {
                const std::size_t team_id = static_cast<std::size_t>(omp_get_team_num());
                if (team_id < pool_size) {
                    const std::size_t v = static_cast<std::size_t>(pool_ptr[team_id]);
                    const std::size_t e_begin = vertex_offsets[v];
                    const std::size_t e_end = vertex_offsets[v + 1];
                    const std::size_t degree = e_end - e_begin;
                    float counts[MAX_LABELS_CAP] = {0.0f};
#pragma omp parallel for reduction(+ : counts[:MAX_LABELS_CAP])
                    for (std::size_t idx = 0; idx < degree; ++idx) {
                        const auto edge = vertex_edges[e_begin + idx];
                        const int lab = static_cast<int>(elabels[edge]);
                        if (lab >= 0 && lab < max_labels) { counts[lab] += 1.0f; }
                    }
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
                        team_changes_ptr[team_id] = 1;
                    }
                }
            }
            std::size_t local_changes = 0;
            for (std::size_t i = 0; i < pool_size; ++i) { local_changes += team_changes[i]; }
            return local_changes;
        };

        auto run_vertex_pool_wi = [&](const std::vector<std::uint32_t>& pool_vertices) -> std::size_t {
            if (pool_vertices.empty()) { return 0; }
            const std::size_t pool_size = pool_vertices.size();
            const std::uint32_t* pool_ptr = pool_vertices.data();
            std::size_t local_changes = 0;
#pragma omp target teams distribute parallel for map(present, to : vertex_edges[0 : vertex_edges_size], vertex_offsets[0 : vertex_offsets_size]) map(present, tofrom : vlabels[0 : num_vertices], elabels[0 : num_edges]) map(to : pool_ptr[0 : pool_size]) reduction(+ : local_changes)
            for (std::size_t idx = 0; idx < pool_size; ++idx) {
                const std::size_t v = static_cast<std::size_t>(pool_ptr[idx]);
                const std::size_t e_begin = vertex_offsets[v];
                const std::size_t e_end = vertex_offsets[v + 1];
                float counts[MAX_LABELS_CAP] = {0.0f};
                for (std::size_t e_idx = e_begin; e_idx < e_end; ++e_idx) {
                    const auto edge = vertex_edges[e_idx];
                    const int lab = static_cast<int>(elabels[edge]);
                    if (lab >= 0 && lab < max_labels) { counts[lab] += 1.0f; }
                }
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
                    local_changes += 1;
                }
            }
            return local_changes;
        };

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            run_edge_pool_wg(exec_pool.wg_pool_edges, wgs);
            run_edge_pool_wi(exec_pool.wi_pool_edges);

            std::size_t changes = 0;
            changes += run_vertex_pool_wg(exec_pool.wg_pool_vertices, wgs);
            changes += run_vertex_pool_wi(exec_pool.wi_pool_vertices);

            const double change_ratio = static_cast<double>(changes) / static_cast<double>(num_vertices);
            if (change_ratio < tolerance) {
                std::cout << "Converged after " << iteration + 1 << " iterations\n";
                converged = true;
                iterations_completed = iteration + 1;
                break;
            }
            if ((iteration + 1) % 10 == 0) { std::cout << "Iteration " << iteration + 1 << " completed\n"; }
        }
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
