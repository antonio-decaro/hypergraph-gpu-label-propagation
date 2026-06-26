#include "pagerank_sycl.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

PageRankSYCL::PageRankSYCL(const CLI::DeviceOptions& device, const sycl::queue& queue, double alpha)
    : PageRankAlgorithm(device, alpha), queue_(queue) {
    std::cout << "SYCL device: " << queue_.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "SYCL platform: " << queue_.get_device().get_platform().get_info<sycl::info::platform::name>() << "\n";
}

PageRankSYCL::~PageRankSYCL() = default;

PerformanceMeasurer PageRankSYCL::run(const Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running SYCL PageRank (alpha=" << alpha_ << ")\n";

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

    DeviceFlatHypergraph hg{};
    float* d_p    = nullptr;
    float* d_h    = nullptr;
    float* d_diff = nullptr;

    const auto setup_start = PerformanceMeasurer::clock::now();

    hg     = create_device_hypergraph(hypergraph);
    d_p    = sycl::malloc_device<float>(num_vertices, queue_);
    d_h    = sycl::malloc_device<float>(num_edges,    queue_);
    d_diff = sycl::malloc_device<float>(1,            queue_);

    const float inv_n = 1.0f / static_cast<float>(num_vertices);
    std::vector<float> h_p(num_vertices, inv_n);
    queue_.copy(h_p.data(), d_p, num_vertices).wait();

    const auto setup_end = PerformanceMeasurer::clock::now();
    perf.add_moment("setup", setup_end - setup_start);

    int iterations_completed = 0;
    bool converged = false;

    try {
        // First iteration timed separately: isolates JIT compilation overhead
        const auto init_start = PerformanceMeasurer::clock::now();
        if (max_iterations > 0) {
            converged = run_iteration_sycl(hg, d_p, d_h, d_diff, tolerance);
            if (converged) {
                std::cout << "Converged after 1 iterations\n";
                iterations_completed = 1;
            }
        }
        const auto init_end = PerformanceMeasurer::clock::now();
        perf.add_moment("init", init_end - init_start);

        const auto iteration_start = PerformanceMeasurer::clock::now();
        for (int iteration = 1; !converged && iteration < max_iterations; ++iteration) {
            converged = run_iteration_sycl(hg, d_p, d_h, d_diff, tolerance);
            if (converged) {
                std::cout << "Converged after " << iteration + 1 << " iterations\n";
                iterations_completed = iteration + 1;
            }
            if (!converged && (iteration + 1) % 10 == 0) { std::cout << "Iteration " << iteration + 1 << " completed\n"; }
        }
        if (!converged) { iterations_completed = max_iterations; }

        const auto iteration_end = PerformanceMeasurer::clock::now();
        perf.add_moment("iterations", iteration_end - iteration_start);

        const auto finalize_start = PerformanceMeasurer::clock::now();
        scores_.resize(num_vertices);
        queue_.copy(d_p, scores_.data(), num_vertices).wait();
        const auto finalize_end = PerformanceMeasurer::clock::now();
        perf.add_moment("finalize", finalize_end - finalize_start);

        perf.set_iterations(iterations_completed);
        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);

        cleanup_flat_hypergraph(hg);
        if (d_p)    { sycl::free(d_p,    queue_); }
        if (d_h)    { sycl::free(d_h,    queue_); }
        if (d_diff) { sycl::free(d_diff, queue_); }

        return perf;
    } catch (...) {
        cleanup_flat_hypergraph(hg);
        if (d_p)    { sycl::free(d_p,    queue_); }
        if (d_h)    { sycl::free(d_h,    queue_); }
        if (d_diff) { sycl::free(d_diff, queue_); }
        throw;
    }
}

bool PageRankSYCL::run_iteration_sycl(
    const DeviceFlatHypergraph& flat_hg, float* d_p, float* d_h, float* d_diff, double tolerance)
{
    const std::size_t wgs   = device_.workgroup_size;
    const float alpha_f     = static_cast<float>(alpha_);
    const float inv_n       = 1.0f / static_cast<float>(flat_hg.num_vertices);

    const Hypergraph::VertexId* edge_vertices  = flat_hg.edge_vertices;
    const std::size_t* edge_offsets            = flat_hg.edge_offsets;
    const Hypergraph::EdgeId* vertex_edges     = flat_hg.vertex_edges;
    const std::size_t* vertex_offsets          = flat_hg.vertex_offsets;
    const std::size_t* edge_sizes              = flat_hg.edge_sizes;

    // Reset diff
    queue_.fill(d_diff, 0.0f, 1).wait();

    // Edge phase: h[e] = sum_{u in e} p[u] / d(u)
    const std::size_t global_e = ((flat_hg.num_edges + wgs - 1) / wgs) * wgs;
    auto edge_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global_e, wgs), [=](sycl::nd_item<1> idx) {
            const std::size_t e = idx.get_global_id(0);
            if (e >= flat_hg.num_edges) return;

            float acc = 0.0f;
            for (std::size_t i = edge_offsets[e]; i < edge_offsets[e + 1]; ++i) {
                const std::size_t u = edge_vertices[i];
                const float du      = static_cast<float>(vertex_offsets[u + 1] - vertex_offsets[u]);
                if (du > 0.0f) acc += d_p[u] / du;
            }
            d_h[e] = acc;
        });
    });

    // Vertex phase: p[v] = (1-alpha)/|V| + alpha * sum_{e in v} h[e] / delta(e)
    const std::size_t global_v = ((flat_hg.num_vertices + wgs - 1) / wgs) * wgs;
    queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(edge_event);
        auto diff_redn = sycl::reduction(d_diff, sycl::plus<float>());
        cgh.parallel_for(sycl::nd_range<1>(global_v, wgs), diff_redn, [=](sycl::nd_item<1> idx, auto& diff_arg) {
            const std::size_t v = idx.get_global_id(0);
            if (v >= flat_hg.num_vertices) return;

            float acc = 0.0f;
            for (std::size_t i = vertex_offsets[v]; i < vertex_offsets[v + 1]; ++i) {
                const std::size_t e = vertex_edges[i];
                const float delta_e = static_cast<float>(edge_sizes[e]);
                if (delta_e > 0.0f) acc += d_h[e] / delta_e;
            }
            const float p_new = (1.0f - alpha_f) * inv_n + alpha_f * acc;
            diff_arg          += sycl::fabs(p_new - d_p[v]);
            d_p[v]             = p_new;
        });
    }).wait();

    float diff_val = 0.0f;
    queue_.copy(d_diff, &diff_val, 1).wait();
    return static_cast<double>(diff_val) < tolerance;
}
