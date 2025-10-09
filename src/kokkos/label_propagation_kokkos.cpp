#include "label_propagation_kokkos.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

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
    constexpr int MAX_LABELS = 10; // keep consistent with SYCL/OpenMP
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // Use team scratch (shared) memory to hold per-thread label counts
        using ExecSpace = ExecutionSpace;
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
        using Member = TeamPolicy::member_type;
        using ScratchSpace = typename ExecSpace::scratch_memory_space;

        const int team_size = device_.workgroup_size > 0 ? static_cast<int>(device_.workgroup_size) : 256;

        // -----------------------------
        // Phase 1: update edge labels
        // -----------------------------
        {
            const std::size_t league_size = (kokkos_hg.num_edges + team_size - 1) / team_size;
            TeamPolicy policy(static_cast<int>(league_size), team_size);
            const std::size_t shmem_bytes = Kokkos::View<float*, ScratchSpace>::shmem_size(MAX_LABELS * team_size);
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_bytes));

            Kokkos::parallel_for(
                "edge_update_shared",
                policy,
                KOKKOS_LAMBDA(const Member& team) {
                    // Allocate per-team scratch buffer sized for (team_size x MAX_LABELS)
                    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged> label_weights(team.team_scratch(0), MAX_LABELS * team.team_size());

                    // Each thread in the team processes one edge
                    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team.team_size()), [&](const int tid) {
                        const std::size_t e = static_cast<std::size_t>(team.league_rank()) * static_cast<std::size_t>(team.team_size()) + static_cast<std::size_t>(tid);
                        if (e >= kokkos_hg.num_edges) return;

                        // Zero per-thread label counters in shared memory
                        const int base = tid * MAX_LABELS;
                        for (int i = 0; i < MAX_LABELS; ++i) { label_weights(base + i) = 0.0f; }

                        const std::size_t v_begin = kokkos_hg.edge_offsets(e);
                        const std::size_t v_end = kokkos_hg.edge_offsets(e + 1);
                        for (std::size_t j = v_begin; j < v_end; ++j) {
                            const auto u = kokkos_hg.edge_vertices(j);
                            const int lab = static_cast<int>(vertex_labels(u));
                            if (lab >= 0 && lab < MAX_LABELS) { label_weights(base + lab) += 1.0f; }
                        }

                        int best = edge_labels(e);
                        float best_w = -1.0f;
                        for (int lab = 0; lab < MAX_LABELS; ++lab) {
                            const float w = label_weights(base + lab);
                            if (w > best_w) {
                                best_w = w;
                                best = lab;
                            }
                        }
                        edge_labels(e) = static_cast<Hypergraph::Label>(best);
                    });
                });
        }

        // ---------------------------------------
        // Phase 2: update vertex labels + reduce
        // ---------------------------------------
        std::size_t changes = 0;
        {
            const std::size_t league_size = (kokkos_hg.num_vertices + team_size - 1) / team_size;
            TeamPolicy policy(static_cast<int>(league_size), team_size);
            const std::size_t shmem_bytes = Kokkos::View<float*, ScratchSpace>::shmem_size(MAX_LABELS * team_size);
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_bytes));

            Kokkos::parallel_reduce(
                "vertex_update_shared",
                policy,
                KOKKOS_LAMBDA(const Member& team, std::size_t& team_changes) {
                    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryUnmanaged> label_weights(team.team_scratch(0), MAX_LABELS * team.team_size());

                    // Each thread processes one vertex; accumulate changes via TeamThreadRange reduction
                    std::size_t local_changes = 0;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, team.team_size()),
                        [&](const int tid, std::size_t& lsum) {
                            const std::size_t v = static_cast<std::size_t>(team.league_rank()) * static_cast<std::size_t>(team.team_size()) + static_cast<std::size_t>(tid);
                            if (v >= kokkos_hg.num_vertices) return;

                            const int base = tid * MAX_LABELS;
                            for (int i = 0; i < MAX_LABELS; ++i) { label_weights(base + i) = 0.0f; }

                            const std::size_t e_begin = kokkos_hg.vertex_offsets(v);
                            const std::size_t e_end = kokkos_hg.vertex_offsets(v + 1);
                            for (std::size_t i = e_begin; i < e_end; ++i) {
                                const auto e = kokkos_hg.vertex_edges(i);
                                const int lab = static_cast<int>(edge_labels(e));
                                if (lab >= 0 && lab < MAX_LABELS) { label_weights(base + lab) += 1.0f; }
                            }

                            int best = vertex_labels(v);
                            float best_w = -1.0f;
                            for (int lab = 0; lab < MAX_LABELS; ++lab) {
                                const float w = label_weights(base + lab);
                                if (w > best_w) {
                                    best_w = w;
                                    best = lab;
                                }
                            }

                            if (vertex_labels(v) != static_cast<Hypergraph::Label>(best)) {
                                vertex_labels(v) = static_cast<Hypergraph::Label>(best);
                                lsum += 1;
                            }
                        },
                        local_changes);

                    // Contribute this team's changes to the global reduction
                    team_changes += local_changes;
                },
                changes);
        }

        const double change_ratio = static_cast<double>(changes) / static_cast<double>(kokkos_hg.num_vertices);
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
