#include "pagerank_kokkos.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

PageRankKokkos::PageRankKokkos(const CLI::DeviceOptions& device, double alpha)
    : PageRankAlgorithm(device, alpha), kokkos_initialized_(false) {
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        kokkos_initialized_ = true;
        std::cout << "Kokkos initialized\n";
    }
    std::cout << "Kokkos execution space: " << typeid(ExecutionSpace).name() << "\n";
    std::cout << "Kokkos memory space: " << typeid(MemorySpace).name() << "\n";
}

PageRankKokkos::~PageRankKokkos() {
    if (kokkos_initialized_ && Kokkos::is_initialized()) { Kokkos::finalize(); }
}

PerformanceMeasurer PageRankKokkos::run(const Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running Kokkos PageRank (alpha=" << alpha_ << ")\n";

    PerformanceMeasurer perf;
    const auto overall_start = PerformanceMeasurer::clock::now();

    if (hypergraph.get_num_vertices() == 0 || hypergraph.get_num_edges() == 0) {
        std::cout << "Empty hypergraph detected; nothing to compute.\n";
        perf.set_iterations(0);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
        return perf;
    }

    const auto setup_start = PerformanceMeasurer::clock::now();

    auto kokkos_hg = create_kokkos_hypergraph(hypergraph);

    FloatView p("p", kokkos_hg.num_vertices);
    FloatView h("h", kokkos_hg.num_edges);

    const float inv_n = 1.0f / static_cast<float>(kokkos_hg.num_vertices);
    Kokkos::deep_copy(p, inv_n);
    Kokkos::deep_copy(h, 0.0f);

    const auto setup_end = PerformanceMeasurer::clock::now();
    perf.add_moment("setup", setup_end - setup_start);

    const auto iteration_start = PerformanceMeasurer::clock::now();
    int iterations_completed = 0;
    bool converged = false;

    const float alpha_f = static_cast<float>(alpha_);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        // Edge phase: h[e] = sum_{u in e} p[u] / d(u)
        Kokkos::parallel_for(
            "pagerank_edge",
            kokkos_hg.num_edges,
            KOKKOS_LAMBDA(const std::size_t e) {
                float acc = 0.0f;
                for (std::size_t i = kokkos_hg.edge_offsets(e); i < kokkos_hg.edge_offsets(e + 1); ++i) {
                    const std::size_t u = kokkos_hg.edge_vertices(i);
                    const float du = static_cast<float>(kokkos_hg.vertex_offsets(u + 1) - kokkos_hg.vertex_offsets(u));
                    if (du > 0.0f) acc += p(u) / du;
                }
                h(e) = acc;
            });
        Kokkos::fence();

        // Vertex phase: p[v] = (1-alpha)/|V| + alpha * sum_{e in v} h[e] / delta(e)
        float diff = 0.0f;
        Kokkos::parallel_reduce(
            "pagerank_vertex",
            kokkos_hg.num_vertices,
            KOKKOS_LAMBDA(const std::size_t v, float& lsum) {
                float acc = 0.0f;
                for (std::size_t i = kokkos_hg.vertex_offsets(v); i < kokkos_hg.vertex_offsets(v + 1); ++i) {
                    const std::size_t e = kokkos_hg.vertex_edges(i);
                    const float delta_e = static_cast<float>(kokkos_hg.edge_sizes(e));
                    if (delta_e > 0.0f) acc += h(e) / delta_e;
                }
                const float p_new = (1.0f - alpha_f) * inv_n + alpha_f * acc;
                lsum += Kokkos::abs(p_new - p(v));
                p(v) = p_new;
            },
            diff);

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
    scores_.resize(kokkos_hg.num_vertices);
    auto h_p = Kokkos::create_mirror_view(p);
    Kokkos::deep_copy(h_p, p);
    for (std::size_t v = 0; v < kokkos_hg.num_vertices; ++v) { scores_[v] = h_p(v); }
    const auto finalize_end = PerformanceMeasurer::clock::now();
    perf.add_moment("finalize", finalize_end - finalize_start);

    perf.set_iterations(iterations_completed);
    perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
    return perf;
}

PageRankKokkos::KokkosHypergraph PageRankKokkos::create_kokkos_hypergraph(const Hypergraph& hypergraph) {
    KokkosHypergraph kg;
    kg.num_vertices = hypergraph.get_num_vertices();
    kg.num_edges    = hypergraph.get_num_edges();

    std::size_t total_ev = 0, total_ve = 0;
    for (std::size_t e = 0; e < kg.num_edges;    ++e) total_ev += hypergraph.get_hyperedge(e).size();
    for (std::size_t v = 0; v < kg.num_vertices; ++v) total_ve += hypergraph.get_incident_edges(v).size();

    kg.edge_vertices  = VertexView("ev", total_ev);
    kg.edge_offsets   = SizeView("eo",  kg.num_edges + 1);
    kg.vertex_edges   = EdgeView("ve",  total_ve);
    kg.vertex_offsets = SizeView("vo",  kg.num_vertices + 1);
    kg.edge_sizes     = SizeView("es",  kg.num_edges);

    auto h_ev = Kokkos::create_mirror_view(kg.edge_vertices);
    auto h_eo = Kokkos::create_mirror_view(kg.edge_offsets);
    auto h_ve = Kokkos::create_mirror_view(kg.vertex_edges);
    auto h_vo = Kokkos::create_mirror_view(kg.vertex_offsets);
    auto h_es = Kokkos::create_mirror_view(kg.edge_sizes);

    std::size_t ev_idx = 0;
    h_eo(0) = 0;
    for (std::size_t e = 0; e < kg.num_edges; ++e) {
        const auto& verts = hypergraph.get_hyperedge(e);
        h_es(e) = verts.size();
        for (auto v : verts) h_ev(ev_idx++) = v;
        h_eo(e + 1) = ev_idx;
    }

    std::size_t ve_idx = 0;
    h_vo(0) = 0;
    for (std::size_t v = 0; v < kg.num_vertices; ++v) {
        const auto& edges = hypergraph.get_incident_edges(v);
        for (auto e : edges) h_ve(ve_idx++) = e;
        h_vo(v + 1) = ve_idx;
    }

    Kokkos::deep_copy(kg.edge_vertices,  h_ev);
    Kokkos::deep_copy(kg.edge_offsets,   h_eo);
    Kokkos::deep_copy(kg.vertex_edges,   h_ve);
    Kokkos::deep_copy(kg.vertex_offsets, h_vo);
    Kokkos::deep_copy(kg.edge_sizes,     h_es);
    return kg;
}
