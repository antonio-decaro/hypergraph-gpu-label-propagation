#include "label_propagation_kokkos.hpp"
#include <Kokkos_Array.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

LabelPropagationKokkos::LabelPropagationKokkos(const CLI::DeviceOptions& device) : LabelPropagationAlgorithm(device), kokkos_initialized_(false) {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        kokkos_initialized_ = true;
        std::cout << "Kokkos initialized\n";
    }

    std::cout << "Kokkos execution space: " << typeid(ExecutionSpace).name() << "\n";
    std::cout << "Kokkos memory space: " << typeid(MemorySpace).name() << "\n";
}

LabelPropagationKokkos::~LabelPropagationKokkos() {
    if (kokkos_initialized_ && Kokkos::is_initialized()) { Kokkos::finalize(); }
}

PerformanceMeasurer LabelPropagationKokkos::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running Kokkos label propagation\n";

    PerformanceMeasurer perf;
    const auto overall_start = PerformanceMeasurer::clock::now();

    if (hypergraph.get_num_vertices() == 0 || hypergraph.get_num_edges() == 0) {
        std::cout << "Empty hypergraph detected; nothing to compute.\n";
        perf.set_iterations(0);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
        return perf;
    }

    const auto setup_start = PerformanceMeasurer::clock::now();
    // Convert to Kokkos representation
    auto kokkos_hg = create_kokkos_hypergraph(hypergraph);

    // Initialize vertex and edge labels
    auto host_init_labels = hypergraph.get_labels();
    LabelView vertex_labels("vertex_labels", host_init_labels.size());
    LabelView edge_labels("edge_labels", kokkos_hg.num_edges);

    // Copy initial vertex labels to device
    {
        auto h_vlabels = Kokkos::create_mirror_view(vertex_labels);
        for (std::size_t i = 0; i < host_init_labels.size(); ++i) { h_vlabels(i) = host_init_labels[i]; }
        Kokkos::deep_copy(vertex_labels, h_vlabels);
    }
    // Initialize edge labels to zero
    Kokkos::deep_copy(edge_labels, Hypergraph::Label(0));

    const auto setup_end = PerformanceMeasurer::clock::now();
    perf.add_moment("setup", setup_end - setup_start);

    const auto iteration_start = PerformanceMeasurer::clock::now();
    int iterations_completed = 0;
    bool converged = false;
    const int max_labels = static_cast<int>(device_.max_labels);
    if (max_labels <= 0) { throw std::invalid_argument("device.max_labels must be > 0"); }
    constexpr int MAX_LABELS_CAP = 32;
    if (max_labels > MAX_LABELS_CAP) { throw std::invalid_argument("device.max_labels must be <= MAX_LABELS_CAP (32) for the Kokkos backend"); }
    auto exec_pool = create_execution_pool(hypergraph);

    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using Member = TeamPolicy::member_type;
    using ScratchSpace = typename ExecutionSpace::scratch_memory_space;

    const int wg_team_size = device_.workgroup_size > 0 ? static_cast<int>(device_.workgroup_size) : -1;
    const int sg_vector_length = 32;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        auto run_edge_pool_team = [&](const char* label, const Kokkos::View<std::uint32_t*, MemorySpace>& pool_view, std::size_t pool_size, int team_size) {
            if (pool_size == 0) { return; }
            TeamPolicy policy(static_cast<int>(pool_size), Kokkos::AUTO);
            if (team_size > 0) { policy = TeamPolicy(static_cast<int>(pool_size), team_size); }
            const std::size_t shmem_bytes = Kokkos::View<float*, ScratchSpace>::shmem_size(static_cast<std::size_t>(max_labels));
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_bytes));

            Kokkos::parallel_for(
                label, policy, KOKKOS_LAMBDA(const Member& team) {
                    const std::size_t pool_index = static_cast<std::size_t>(team.league_rank());
                    if (pool_index >= pool_size) { return; }

                    const Hypergraph::EdgeId edge = static_cast<Hypergraph::EdgeId>(pool_view(pool_index));
                    if (edge >= kokkos_hg.num_edges) { return; }

                    const std::size_t vertex_begin = kokkos_hg.edge_offsets(edge);
                    const std::size_t vertex_end = kokkos_hg.edge_offsets(edge + 1);
                    const std::size_t degree = vertex_end - vertex_begin;

                    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged> label_weights(team.team_scratch(0), max_labels);

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, max_labels), [&](int lab) { label_weights(lab) = 0.0f; });
                    team.team_barrier();

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, static_cast<int>(degree)), [&](int offset) {
                        const auto vertex = kokkos_hg.edge_vertices(vertex_begin + static_cast<std::size_t>(offset));
                        const int lab = static_cast<int>(vertex_labels(vertex));
                        if (lab >= 0 && lab < max_labels) { Kokkos::atomic_add(&label_weights(lab), 1.0f); }
                    });
                    team.team_barrier();

                    if (team.team_rank() == 0) {
                        int best_label = edge_labels(edge);
                        float max_weight = -1.0f;
                        for (int lab = 0; lab < max_labels; ++lab) {
                            const float w = label_weights(lab);
                            if (w > max_weight) {
                                max_weight = w;
                                best_label = lab;
                            }
                        }
                        edge_labels(edge) = static_cast<Hypergraph::Label>(best_label);
                    }
                });
        };

        auto run_edge_pool_vector = [&](const char* label, const Kokkos::View<std::uint32_t*, MemorySpace>& pool_view, std::size_t pool_size, int vector_length) {
            if (pool_size == 0) { return; }
            TeamPolicy policy(static_cast<int>(pool_size), 1, vector_length);
            const std::size_t shmem_bytes = Kokkos::View<float*, ScratchSpace>::shmem_size(static_cast<std::size_t>(max_labels));
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_bytes));

            Kokkos::parallel_for(
                label, policy, KOKKOS_LAMBDA(const Member& team) {
                    const std::size_t pool_index = static_cast<std::size_t>(team.league_rank());
                    if (pool_index >= pool_size) { return; }

                    const Hypergraph::EdgeId edge = static_cast<Hypergraph::EdgeId>(pool_view(pool_index));
                    if (edge >= kokkos_hg.num_edges) { return; }
                    const std::size_t vertex_begin = kokkos_hg.edge_offsets(edge);
                    const std::size_t vertex_end = kokkos_hg.edge_offsets(edge + 1);

                    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged> label_weights(team.team_scratch(0), max_labels);
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, max_labels), [&](int lab) { label_weights(lab) = 0.0f; });
                    team.team_barrier();

                    const int degree = static_cast<int>(vertex_end - vertex_begin);
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, degree), [&](int offset) {
                        const auto vertex = kokkos_hg.edge_vertices(vertex_begin + static_cast<std::size_t>(offset));
                        const int lab = static_cast<int>(vertex_labels(vertex));
                        if (lab >= 0 && lab < max_labels) { Kokkos::atomic_add(&label_weights(lab), 1.0f); }
                    });
                    team.team_barrier();

                    if (team.team_rank() == 0) {
                        int best_label = edge_labels(edge);
                        float best_weight = -1.0f;
                        for (int lab = 0; lab < max_labels; ++lab) {
                            const float w = label_weights(lab);
                            if (w > best_weight) {
                                best_weight = w;
                                best_label = lab;
                            }
                        }
                        edge_labels(edge) = static_cast<Hypergraph::Label>(best_label);
                    }
                });
        };

        auto run_edge_pool_wi = [&](const char* label, const Kokkos::View<std::uint32_t*, MemorySpace>& pool_view, std::size_t pool_size) {
            if (pool_size == 0) { return; }
            Kokkos::parallel_for(
                label, Kokkos::RangePolicy<ExecutionSpace, std::size_t>(0, pool_size), KOKKOS_LAMBDA(const std::size_t idx) {
                    const Hypergraph::EdgeId edge = static_cast<Hypergraph::EdgeId>(pool_view(idx));
                    if (edge >= kokkos_hg.num_edges) { return; }
                    const std::size_t vertex_begin = kokkos_hg.edge_offsets(edge);
                    const std::size_t vertex_end = kokkos_hg.edge_offsets(edge + 1);

                    Kokkos::Array<float, MAX_LABELS_CAP> counts;
                    for (int lab = 0; lab < max_labels; ++lab) { counts[lab] = 0.0f; }

                    for (std::size_t v_idx = vertex_begin; v_idx < vertex_end; ++v_idx) {
                        const auto vertex = kokkos_hg.edge_vertices(v_idx);
                        const int lab = static_cast<int>(vertex_labels(vertex));
                        if (lab >= 0 && lab < max_labels) { counts[lab] += 1.0f; }
                    }

                    int best_label = edge_labels(edge);
                    float best_weight = -1.0f;
                    for (int lab = 0; lab < max_labels; ++lab) {
                        const float w = counts[lab];
                        if (w > best_weight) {
                            best_weight = w;
                            best_label = lab;
                        }
                    }
                    edge_labels(edge) = static_cast<Hypergraph::Label>(best_label);
                });
        };

        run_edge_pool_team("edge_update_wg_pool", exec_pool.wg_pool_edges, exec_pool.wg_pool_edges_size, wg_team_size);
        run_edge_pool_vector("edge_update_sg_pool", exec_pool.sg_pool_edges, exec_pool.sg_pool_edges_size, sg_vector_length);
        run_edge_pool_wi("edge_update_wi_pool", exec_pool.wi_pool_edges, exec_pool.wi_pool_edges_size);

        std::size_t iteration_changes = 0;
        auto run_vertex_pool_team = [&](const char* label, const Kokkos::View<std::uint32_t*, MemorySpace>& pool_view, std::size_t pool_size, int team_size) -> std::size_t {
            if (pool_size == 0) { return 0; }
            TeamPolicy policy(static_cast<int>(pool_size), Kokkos::AUTO);
            if (team_size > 0) { policy = TeamPolicy(static_cast<int>(pool_size), team_size); }
            const std::size_t shmem_bytes = Kokkos::View<float*, ScratchSpace>::shmem_size(static_cast<std::size_t>(max_labels));
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_bytes));

            std::size_t pool_changes = 0;
            Kokkos::parallel_reduce(
                label,
                policy,
                KOKKOS_LAMBDA(const Member& team, std::size_t& local_changes) {
                    const std::size_t pool_index = static_cast<std::size_t>(team.league_rank());
                    if (pool_index >= pool_size) { return; }

                    const Hypergraph::VertexId vertex = static_cast<Hypergraph::VertexId>(pool_view(pool_index));
                    if (vertex >= kokkos_hg.num_vertices) { return; }

                    const std::size_t edge_begin = kokkos_hg.vertex_offsets(vertex);
                    const std::size_t edge_end = kokkos_hg.vertex_offsets(vertex + 1);
                    const std::size_t incident_degree = edge_end - edge_begin;

                    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged> label_weights(team.team_scratch(0), max_labels);
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, max_labels), [&](int lab) { label_weights(lab) = 0.0f; });
                    team.team_barrier();

                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, static_cast<int>(incident_degree)), [&](int offset) {
                        const auto edge = kokkos_hg.vertex_edges(edge_begin + static_cast<std::size_t>(offset));
                        const int lab = static_cast<int>(edge_labels(edge));
                        if (lab >= 0 && lab < max_labels) { Kokkos::atomic_add(&label_weights(lab), 1.0f); }
                    });
                    team.team_barrier();

                    if (team.team_rank() == 0) {
                        int best_label = vertex_labels(vertex);
                        float max_weight = -1.0f;
                        for (int lab = 0; lab < max_labels; ++lab) {
                            const float w = label_weights(lab);
                            if (w > max_weight) {
                                max_weight = w;
                                best_label = lab;
                            }
                        }

                        if (vertex_labels(vertex) != static_cast<Hypergraph::Label>(best_label)) {
                            vertex_labels(vertex) = static_cast<Hypergraph::Label>(best_label);
                            local_changes += 1;
                        }
                    }
                },
                pool_changes);

            return pool_changes;
        };

        auto run_vertex_pool_vector = [&](const char* label, const Kokkos::View<std::uint32_t*, MemorySpace>& pool_view, std::size_t pool_size, int vector_length) -> std::size_t {
            if (pool_size == 0) { return 0; }
            std::size_t pool_changes = 0;
            TeamPolicy policy(static_cast<int>(pool_size), 1, vector_length);
            const std::size_t shmem_bytes = Kokkos::View<float*, ScratchSpace>::shmem_size(static_cast<std::size_t>(max_labels));
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_bytes));

            Kokkos::parallel_reduce(
                label,
                policy,
                KOKKOS_LAMBDA(const Member& team, std::size_t& local_changes) {
                    const std::size_t pool_index = static_cast<std::size_t>(team.league_rank());
                    if (pool_index >= pool_size) { return; }

                    const Hypergraph::VertexId vertex = static_cast<Hypergraph::VertexId>(pool_view(pool_index));
                    if (vertex >= kokkos_hg.num_vertices) { return; }
                    const std::size_t edge_begin = kokkos_hg.vertex_offsets(vertex);
                    const std::size_t edge_end = kokkos_hg.vertex_offsets(vertex + 1);

                    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged> label_weights(team.team_scratch(0), max_labels);
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, max_labels), [&](int lab) { label_weights(lab) = 0.0f; });
                    team.team_barrier();

                    const int incident_degree = static_cast<int>(edge_end - edge_begin);
                    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, incident_degree), [&](int offset) {
                        const auto edge = kokkos_hg.vertex_edges(edge_begin + static_cast<std::size_t>(offset));
                        const int lab = static_cast<int>(edge_labels(edge));
                        if (lab >= 0 && lab < max_labels) { Kokkos::atomic_add(&label_weights(lab), 1.0f); }
                    });
                    team.team_barrier();

                    if (team.team_rank() == 0) {
                        int best_label = vertex_labels(vertex);
                        float best_weight = -1.0f;
                        for (int lab = 0; lab < max_labels; ++lab) {
                            const float w = label_weights(lab);
                            if (w > best_weight) {
                                best_weight = w;
                                best_label = lab;
                            }
                        }

                        if (vertex_labels(vertex) != static_cast<Hypergraph::Label>(best_label)) {
                            vertex_labels(vertex) = static_cast<Hypergraph::Label>(best_label);
                            local_changes += 1;
                        }
                    }
                },
                pool_changes);

            return pool_changes;
        };

        auto run_vertex_pool_wi = [&](const char* label, const Kokkos::View<std::uint32_t*, MemorySpace>& pool_view, std::size_t pool_size) -> std::size_t {
            if (pool_size == 0) { return 0; }
            std::size_t local_changes = 0;
            Kokkos::parallel_reduce(
                label,
                Kokkos::RangePolicy<ExecutionSpace, std::size_t>(0, pool_size),
                KOKKOS_LAMBDA(const std::size_t idx, std::size_t& thread_changes) {
                    const Hypergraph::VertexId vertex = static_cast<Hypergraph::VertexId>(pool_view(idx));
                    if (vertex >= kokkos_hg.num_vertices) { return; }
                    const std::size_t edge_begin = kokkos_hg.vertex_offsets(vertex);
                    const std::size_t edge_end = kokkos_hg.vertex_offsets(vertex + 1);

                    Kokkos::Array<float, MAX_LABELS_CAP> counts;
                    for (int lab = 0; lab < max_labels; ++lab) { counts[lab] = 0.0f; }

                    for (std::size_t e_idx = edge_begin; e_idx < edge_end; ++e_idx) {
                        const auto edge = kokkos_hg.vertex_edges(e_idx);
                        const int lab = static_cast<int>(edge_labels(edge));
                        if (lab >= 0 && lab < max_labels) { counts[lab] += 1.0f; }
                    }

                    int best_label = vertex_labels(vertex);
                    float best_weight = -1.0f;
                    for (int lab = 0; lab < max_labels; ++lab) {
                        const float w = counts[lab];
                        if (w > best_weight) {
                            best_weight = w;
                            best_label = lab;
                        }
                    }

                    if (vertex_labels(vertex) != static_cast<Hypergraph::Label>(best_label)) {
                        vertex_labels(vertex) = static_cast<Hypergraph::Label>(best_label);
                        thread_changes += 1;
                    }
                },
                local_changes);
            return local_changes;
        };

        iteration_changes += run_vertex_pool_team("vertex_update_wg_pool", exec_pool.wg_pool_vertices, exec_pool.wg_pool_vertices_size, wg_team_size);
        iteration_changes += run_vertex_pool_vector("vertex_update_sg_pool", exec_pool.sg_pool_vertices, exec_pool.sg_pool_vertices_size, sg_vector_length);
        iteration_changes += run_vertex_pool_wi("vertex_update_wi_pool", exec_pool.wi_pool_vertices, exec_pool.wi_pool_vertices_size);

        const double change_ratio = static_cast<double>(iteration_changes) / static_cast<double>(kokkos_hg.num_vertices);
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
    // Copy results back to host
    auto host_labels = Kokkos::create_mirror_view(vertex_labels);
    Kokkos::deep_copy(host_labels, vertex_labels);
    std::vector<Hypergraph::Label> final_labels(host_init_labels.size());
    for (std::size_t i = 0; i < host_init_labels.size(); ++i) { final_labels[i] = host_labels(i); }
    hypergraph.set_labels(final_labels);

    const auto finalize_end = PerformanceMeasurer::clock::now();
    perf.add_moment("finalize", finalize_end - finalize_start);

    perf.set_iterations(iterations_completed);
    perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
    return perf;
}

LabelPropagationKokkos::ExecutionPool LabelPropagationKokkos::create_execution_pool(const Hypergraph& hypergraph) {
    ExecutionPool pool{};

    std::vector<std::uint32_t> wg_pool_edges_vec;
    std::vector<std::uint32_t> sg_pool_edges_vec;
    std::vector<std::uint32_t> wi_pool_edges_vec;
    std::vector<std::uint32_t> wg_pool_vertices_vec;
    std::vector<std::uint32_t> sg_pool_vertices_vec;
    std::vector<std::uint32_t> wi_pool_vertices_vec;

    const std::size_t num_edges = hypergraph.get_num_edges();
    const std::size_t num_vertices = hypergraph.get_num_vertices();

    for (std::size_t e = 0; e < num_edges; ++e) {
        const auto edge_size = hypergraph.get_hyperedge(e).size();
        if (edge_size > 1024) {
            wg_pool_edges_vec.push_back(static_cast<std::uint32_t>(e));
        } else if (edge_size > 256) {
            sg_pool_edges_vec.push_back(static_cast<std::uint32_t>(e));
        } else {
            wi_pool_edges_vec.push_back(static_cast<std::uint32_t>(e));
        }
    }

    for (std::size_t v = 0; v < num_vertices; ++v) {
        const auto incident_size = hypergraph.get_incident_edges(v).size();
        if (incident_size > 1024) {
            wg_pool_vertices_vec.push_back(static_cast<std::uint32_t>(v));
        } else if (incident_size > 256) {
            sg_pool_vertices_vec.push_back(static_cast<std::uint32_t>(v));
        } else {
            wi_pool_vertices_vec.push_back(static_cast<std::uint32_t>(v));
        }
    }

    auto copy_pool = [](const std::vector<std::uint32_t>& src, Kokkos::View<std::uint32_t*, MemorySpace>& dst, std::size_t& size, const char* name) {
        size = src.size();
        if (size == 0) { return; }
        dst = Kokkos::View<std::uint32_t*, MemorySpace>(name, size);
        auto host_view = Kokkos::create_mirror_view(dst);
        for (std::size_t i = 0; i < size; ++i) { host_view(i) = src[i]; }
        Kokkos::deep_copy(dst, host_view);
    };

    copy_pool(wg_pool_edges_vec, pool.wg_pool_edges, pool.wg_pool_edges_size, "wg_pool_edges");
    copy_pool(sg_pool_edges_vec, pool.sg_pool_edges, pool.sg_pool_edges_size, "sg_pool_edges");
    copy_pool(wi_pool_edges_vec, pool.wi_pool_edges, pool.wi_pool_edges_size, "wi_pool_edges");
    copy_pool(wg_pool_vertices_vec, pool.wg_pool_vertices, pool.wg_pool_vertices_size, "wg_pool_vertices");
    copy_pool(sg_pool_vertices_vec, pool.sg_pool_vertices, pool.sg_pool_vertices_size, "sg_pool_vertices");
    copy_pool(wi_pool_vertices_vec, pool.wi_pool_vertices, pool.wi_pool_vertices_size, "wi_pool_vertices");

    return pool;
}

LabelPropagationKokkos::KokkosHypergraph LabelPropagationKokkos::create_kokkos_hypergraph(const Hypergraph& hypergraph) {
    KokkosHypergraph kokkos_hg;
    kokkos_hg.num_vertices = hypergraph.get_num_vertices();
    kokkos_hg.num_edges = hypergraph.get_num_edges();

    // Calculate sizes for flattened arrays
    std::size_t total_edge_vertices = 0;
    std::size_t total_vertex_edges = 0;

    for (std::size_t e = 0; e < kokkos_hg.num_edges; ++e) { total_edge_vertices += hypergraph.get_hyperedge(e).size(); }

    for (std::size_t v = 0; v < kokkos_hg.num_vertices; ++v) { total_vertex_edges += hypergraph.get_incident_edges(v).size(); }

    // Allocate Kokkos views
    kokkos_hg.edge_vertices = VertexView("edge_vertices", total_edge_vertices);
    kokkos_hg.edge_offsets = SizeView("edge_offsets", kokkos_hg.num_edges + 1);
    kokkos_hg.vertex_edges = EdgeView("vertex_edges", total_vertex_edges);
    kokkos_hg.vertex_offsets = SizeView("vertex_offsets", kokkos_hg.num_vertices + 1);
    kokkos_hg.edge_sizes = SizeView("edge_sizes", kokkos_hg.num_edges);

    // Create host mirrors and fill data
    auto h_edge_vertices = Kokkos::create_mirror_view(kokkos_hg.edge_vertices);
    auto h_edge_offsets = Kokkos::create_mirror_view(kokkos_hg.edge_offsets);
    auto h_vertex_edges = Kokkos::create_mirror_view(kokkos_hg.vertex_edges);
    auto h_vertex_offsets = Kokkos::create_mirror_view(kokkos_hg.vertex_offsets);
    auto h_edge_sizes = Kokkos::create_mirror_view(kokkos_hg.edge_sizes);

    // Fill edge data
    std::size_t edge_vertex_idx = 0;
    h_edge_offsets(0) = 0;

    for (std::size_t e = 0; e < kokkos_hg.num_edges; ++e) {
        const auto& vertices = hypergraph.get_hyperedge(e);
        h_edge_sizes(e) = vertices.size();

        for (auto v : vertices) { h_edge_vertices(edge_vertex_idx++) = v; }
        h_edge_offsets(e + 1) = edge_vertex_idx;
    }

    // Fill vertex data
    std::size_t vertex_edge_idx = 0;
    h_vertex_offsets(0) = 0;

    for (std::size_t v = 0; v < kokkos_hg.num_vertices; ++v) {
        const auto& edges = hypergraph.get_incident_edges(v);

        for (auto e : edges) { h_vertex_edges(vertex_edge_idx++) = e; }
        h_vertex_offsets(v + 1) = vertex_edge_idx;
    }

    // Copy to device
    Kokkos::deep_copy(kokkos_hg.edge_vertices, h_edge_vertices);
    Kokkos::deep_copy(kokkos_hg.edge_offsets, h_edge_offsets);
    Kokkos::deep_copy(kokkos_hg.vertex_edges, h_vertex_edges);
    Kokkos::deep_copy(kokkos_hg.vertex_offsets, h_vertex_offsets);
    Kokkos::deep_copy(kokkos_hg.edge_sizes, h_edge_sizes);

    return kokkos_hg;
}

// run_iteration_kokkos removed: algorithm is now implemented inline in run() with two phases
