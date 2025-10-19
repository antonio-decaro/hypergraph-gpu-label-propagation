#include "label_propagation_sycl.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

LabelPropagationSYCL::LabelPropagationSYCL(const CLI::DeviceOptions& device, const sycl::queue& queue) : LabelPropagationAlgorithm(device), queue_(queue) {
    std::cout << "SYCL device: " << queue_.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "SYCL platform: " << queue_.get_device().get_platform().get_info<sycl::info::platform::name>() << "\n";
}

LabelPropagationSYCL::~LabelPropagationSYCL() = default;

PerformanceMeasurer LabelPropagationSYCL::run(Hypergraph& hypergraph, int max_iterations, double tolerance) {
    std::cout << "Running SYCL label propagation\n";

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

    const int max_labels = static_cast<int>(device_.max_labels);
    if (max_labels <= 0) { throw std::invalid_argument("device.max_labels must be > 0"); }

    DeviceFlatHypergraph hg{};
    size_t* changes = nullptr;
    Hypergraph::Label* vertex_labels = nullptr;
    Hypergraph::Label* edge_labels = nullptr;

    const auto setup_start = PerformanceMeasurer::clock::now();

    // Flatten the hypergraph for GPU processing
    hg = create_device_hypergraph(hypergraph);
    changes = sycl::malloc_device<size_t>(1, queue_);

    try {
        // Create SYCL arrays for labels
        vertex_labels = sycl::malloc_device<Hypergraph::Label>(num_vertices, queue_);
        edge_labels = sycl::malloc_device<Hypergraph::Label>(num_edges, queue_);

        queue_.copy(hypergraph.get_labels().data(), vertex_labels, num_vertices).wait();

        const auto setup_end = PerformanceMeasurer::clock::now();
        perf.add_moment("setup", setup_end - setup_start);

        auto exec_pool = create_execution_pool(hypergraph);
        const auto init_end = PerformanceMeasurer::clock::now();
        perf.add_moment("init", init_end - setup_end);

        const auto iteration_start = PerformanceMeasurer::clock::now();
        int iterations_completed = 0;
        bool converged = false;
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            bool iteration_converged = run_iteration_sycl(hg, exec_pool, vertex_labels, edge_labels, changes, max_labels, tolerance);

            if (iteration_converged) {
                std::cout << "Converged after " << iteration + 1 << " iterations\n";
                iterations_completed = iteration + 1;
                converged = true;
                break;
            }

            if ((iteration + 1) % 10 == 0) { std::cout << "Iteration " << iteration + 1 << " completed\n"; }
        }

        if (!converged) { iterations_completed = max_iterations; }

        const auto iteration_end = PerformanceMeasurer::clock::now();
        perf.add_moment("iterations", iteration_end - iteration_start);

        const auto finalize_start = PerformanceMeasurer::clock::now();
        // Copy results back
        std::vector<Hypergraph::Label> final_labels(num_vertices);
        queue_.copy(vertex_labels, final_labels.data(), num_vertices).wait();
        hypergraph.set_labels(final_labels);

        const auto finalize_end = PerformanceMeasurer::clock::now();
        perf.add_moment("finalize", finalize_end - finalize_start);
        perf.set_iterations(iterations_completed);

        cleanup_flat_hypergraph(hg);
        if (changes) {
            sycl::free(changes, queue_);
            changes = nullptr;
        }
        if (vertex_labels) {
            sycl::free(vertex_labels, queue_);
            vertex_labels = nullptr;
        }
        if (edge_labels) {
            sycl::free(edge_labels, queue_);
            edge_labels = nullptr;
        }

        perf.set_total_time(PerformanceMeasurer::clock::now() - overall_start);
        return perf;
    } catch (...) {
        cleanup_flat_hypergraph(hg);
        if (changes) { sycl::free(changes, queue_); }
        if (vertex_labels) { sycl::free(vertex_labels, queue_); }
        if (edge_labels) { sycl::free(edge_labels, queue_); }
        throw;
    }
}

bool LabelPropagationSYCL::run_iteration_sycl(
    const DeviceFlatHypergraph& flat_hg, const DeviceExecutionPool& pool, Hypergraph::Label* vertex_labels, Hypergraph::Label* edge_labels, std::size_t* changes, int max_labels, double tolerance) {
    size_t workgroup_size = device_.workgroup_size;

    std::vector<sycl::event> events;

    if (max_labels <= 0) { throw std::invalid_argument("device.max_labels must be > 0"); }

    // Submit label propagation kernel
    try {
        Hypergraph::VertexId* edge_vertices = flat_hg.edge_vertices;
        std::size_t* edge_offsets = flat_hg.edge_offsets;
        Hypergraph::EdgeId* vertex_edges = flat_hg.vertex_edges;
        std::size_t* vertex_offsets = flat_hg.vertex_offsets;
        std::size_t* edge_sizes = flat_hg.edge_sizes;
        std::size_t* changes_ptr = changes;

        if (pool.wg_pool_edges_size > 0) {
            auto e_wgreduce = queue_.submit([&](sycl::handler& h) {
                if (!events.empty()) { h.depends_on(events.back()); }
                size_t global_size = pool.wg_pool_edges_size * workgroup_size;

                sycl::local_accessor<float, 1> label_weights(static_cast<std::size_t>(max_labels), h);

                h.parallel_for(sycl::nd_range<1>(global_size, workgroup_size), [=](sycl::nd_item<1> idx) {
                    const auto edge_id = idx.get_group_linear_id();
                    const auto edge = pool.wg_pool_edges[edge_id];
                    const auto degree = edge_offsets[edge + 1] - edge_offsets[edge];

                    if (idx.get_local_linear_id() < max_labels) label_weights[idx.get_local_linear_id()] = 0.0f;
                    sycl::group_barrier(idx.get_group());

                    for (auto vertex_id = idx.get_local_linear_id(); vertex_id < degree; vertex_id += idx.get_group_range(0)) {
                        auto vertex = edge_vertices[edge_offsets[edge] + vertex_id];
                        auto label = vertex_labels[vertex];
                        if (label >= 0 && label < max_labels) {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group> label_aref(label_weights[label]);
                            label_aref.fetch_add(1.0f);
                        }
                    }

                    sycl::group_barrier(idx.get_group());

                    if (idx.get_group().leader()) {
                        auto best_label = edge_labels[edge];
                        float max_weight = -1.0f;

                        for (auto label = 0; label < max_labels; ++label) {
                            const float w = label_weights[label];
                            if (w > max_weight) {
                                max_weight = w;
                                best_label = label;
                            }
                        }
                        edge_labels[edge] = best_label;
                    }
                });
            });
            events.push_back(e_wgreduce);
        }

        if (pool.sg_pool_edges_size > 0) {
            auto e_sgreduce = queue_.submit([&](sycl::handler& h) {
                if (!events.empty()) { h.depends_on(events.back()); }
                const size_t sg_size = 32;
                const size_t sg_per_wg = workgroup_size / sg_size;
                size_t global_size = ((pool.sg_pool_edges_size / sg_per_wg) + ((pool.sg_pool_edges_size % sg_per_wg) ? 1 : 0)) * workgroup_size;

                sycl::local_accessor<float, 1> label_weights(static_cast<std::size_t>(max_labels * sg_per_wg), h);

                h.parallel_for(sycl::nd_range<1>(global_size, workgroup_size), [=](sycl::nd_item<1> idx) {
                    const auto sg_id = idx.get_sub_group().get_group_linear_id();
                    const auto edge_id = sg_id + (idx.get_group_linear_id() * sg_per_wg);
                    const auto edge = pool.sg_pool_edges[edge_id];
                    const auto degree = edge_offsets[edge + 1] - edge_offsets[edge];

                    for (auto vertex_id = idx.get_local_linear_id(); vertex_id < degree; vertex_id += idx.get_group_range(0)) {
                        auto vertex = edge_vertices[edge_offsets[edge] + vertex_id];
                        auto label = vertex_labels[vertex];
                        if (label >= 0 && label < max_labels) {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> label_aref(label_weights[label * sg_per_wg + sg_id]);
                            label_aref += 1.0f;
                        }
                    }

                    sycl::group_barrier(idx.get_sub_group());

                    if (idx.get_sub_group().leader()) {
                        auto best_label = edge_labels[edge];
                        float max_weight = -1.0f;

                        for (auto label = 0; label < max_labels; ++label) {
                            const float w = label_weights[label * sg_per_wg + sg_id];
                            if (w > max_weight) {
                                max_weight = w;
                                best_label = label;
                            }
                        }
                        edge_labels[edge] = best_label;
                    }
                });
            });
            events.push_back(e_sgreduce);
        }

        // First update edge labels based on vertex labels
        if (pool.wi_pool_edges_size > 0) {
            auto e = queue_.submit([&](sycl::handler& h) {
                if (!events.empty()) { h.depends_on(events.back()); }

                size_t global_size = ((pool.wi_pool_edges_size + workgroup_size - 1) / workgroup_size) * workgroup_size;

                sycl::local_accessor<float, 1> label_weights(static_cast<std::size_t>(max_labels) * workgroup_size, h);

                h.parallel_for(sycl::nd_range<1>(global_size, workgroup_size), [=](sycl::nd_item<1> idx) {
                    const std::size_t edge_id = idx.get_global_id(0);
                    if (edge_id >= pool.wi_pool_edges_size) return;
                    const auto edge = pool.wi_pool_edges[edge_id];
                    const std::size_t lid = idx.get_local_linear_id();

                    if (edge == 0) { *changes_ptr = 0; }

                    if (edge >= flat_hg.num_edges) return;

                    for (int i = 0; i < max_labels; ++i) { label_weights[(lid * max_labels) + i] = 0.0f; }

                    // Get incident vertices for edge_id
                    const std::size_t vertex_start = edge_offsets[edge];
                    const std::size_t vertex_end = edge_offsets[edge + 1];
                    const std::size_t edge_size = vertex_end - vertex_start;

                    // Process each incident vertex
                    for (std::size_t vertex_id = vertex_start; vertex_id < vertex_end; ++vertex_id) {
                        auto vertex = edge_vertices[vertex_id];
                        std::size_t vertices_start = edge_offsets[edge];

                        auto label = vertex_labels[vertex];
                        if (label >= 0 && label < max_labels) { label_weights[(lid * max_labels) + label] += 1.0f; }
                    }

                    // Find label with maximum weight
                    Hypergraph::Label best_label = edge_labels[edge];
                    float max_weight = -1.0f;

                    for (int label = 0; label < max_labels; ++label) {
                        const float w = label_weights[(lid * max_labels) + label];
                        if (w > max_weight) {
                            max_weight = w;
                            best_label = label;
                        }
                    }

                    edge_labels[edge] = best_label;
                });
            });
            events.push_back(e);
        }

        // Then update vertex labels based on edge labels
        queue_
            .submit([&](sycl::handler& h) {
                if (!events.empty()) { h.depends_on(events.back()); }
                size_t global_size = ((flat_hg.num_vertices + workgroup_size - 1) / workgroup_size) * workgroup_size;

                sycl::local_accessor<float, 1> label_weights(static_cast<std::size_t>(max_labels) * workgroup_size, h);
                auto sumr = sycl::reduction<std::size_t>(changes_ptr, sycl::plus<>());

                h.parallel_for(sycl::nd_range<1>(global_size, workgroup_size), sumr, [=](sycl::nd_item<1> idx, auto& sum_arg) {
                    const std::size_t v = idx.get_global_id(0);
                    const std::size_t lid = idx.get_local_linear_id();

                    if (v >= flat_hg.num_vertices) return;

#pragma unroll
                    for (int i = 0; i < max_labels; ++i) { label_weights[(lid * max_labels) + i] = 0.0f; }

                    // Get incident vertices for edge_id
                    const std::size_t edge_start = vertex_offsets[v];
                    const std::size_t edge_end = vertex_offsets[v + 1];

                    // Process each incident vertex
                    for (std::size_t edge_id = edge_start; edge_id < edge_end; ++edge_id) {
                        auto edge = vertex_edges[edge_id];

                        auto label = edge_labels[edge];
                        if (label >= 0 && label < max_labels) { label_weights[(lid * max_labels) + label] += 1.0f; }
                    }

                    // Find label with maximum weight
                    Hypergraph::Label best_label = vertex_labels[v];
                    float max_weight = -1.0f;

                    for (int label = 0; label < max_labels; ++label) {
                        const float w = label_weights[(lid * max_labels) + label];
                        if (w > max_weight) {
                            max_weight = w;
                            best_label = label;
                        }
                    }

                    // Count changes for convergence check
                    if (vertex_labels[v] != best_label) {
                        sum_arg += 1;
                        vertex_labels[v] = best_label;
                    }
                });
            })
            .wait();

    } catch (...) { throw; }

    // Check convergence
    std::size_t changes_count;
    queue_.copy(changes, &changes_count, 1).wait();
    double change_ratio = static_cast<double>(changes_count) / static_cast<double>(flat_hg.num_vertices);
    return change_ratio < tolerance;
}

LabelPropagationSYCL::DeviceExecutionPool LabelPropagationSYCL::create_execution_pool(const Hypergraph& hypergraph) {
    DeviceExecutionPool pool{};
    // For simplicity, we allocate fixed sizes for each pool based on hypergraph size
    const std::size_t num_edges = hypergraph.get_num_edges();
    const std::size_t num_vertices = hypergraph.get_num_vertices();

    // pool.tmp_buffer_size = std::max(num_edges, num_vertices);
    // pool.tmp_buffer = sycl::malloc_device<std::uint32_t>(pool.tmp_buffer_size, queue_);

    std::vector<std::uint32_t> wg_pool_edges_vec;
    std::vector<std::uint32_t> wg_pool_vertices_vec;
    std::vector<std::uint32_t> sg_pool_edges_vec;
    std::vector<std::uint32_t> sg_pool_vertices_vec;
    std::vector<std::uint32_t> wi_pool_edges_vec;
    std::vector<std::uint32_t> wi_pool_vertices_vec;

    for (size_t i = 0; i < num_edges; ++i) {
        if (hypergraph.get_hyperedge(i).size() > 256) {
            wg_pool_edges_vec.push_back(i);
        } else if (hypergraph.get_hyperedge(i).size() > 32) {
            sg_pool_edges_vec.push_back(i);
        } else {
            // wg_pool_edges_vec.push_back(i);
            wi_pool_edges_vec.push_back(i);
        }
    }

    for (size_t i = 0; i < num_vertices; ++i) {
        if (hypergraph.get_incident_edges(i).size() > 1024) {
            wg_pool_vertices_vec.push_back(i);
        } else if (hypergraph.get_incident_edges(i).size() > 256) {
            sg_pool_vertices_vec.push_back(i);
        } else {
            wi_pool_vertices_vec.push_back(i);
        }
    }

    std::cout << "Work-group pool edges: " << wg_pool_edges_vec.size() << "\n";
    std::cout << "Work-group pool vertices: " << wg_pool_vertices_vec.size() << "\n";
    std::cout << "Sub-group pool edges: " << sg_pool_edges_vec.size() << "\n";
    std::cout << "Sub-group pool vertices: " << sg_pool_vertices_vec.size() << "\n";
    std::cout << "Work-item pool edges: " << wi_pool_edges_vec.size() << "\n";
    std::cout << "Work-item pool vertices: " << wi_pool_vertices_vec.size() << "\n";

    pool.wg_pool_edges_size = static_cast<std::uint32_t>(wg_pool_edges_vec.size());
    pool.wg_pool_edges = sycl::malloc_device<std::uint32_t>(pool.wg_pool_edges_size, queue_);
    pool.wg_pool_vertices_size = static_cast<std::uint32_t>(wg_pool_vertices_vec.size());
    pool.wg_pool_vertices = sycl::malloc_device<std::uint32_t>(pool.wg_pool_vertices_size, queue_);
    pool.sg_pool_edges_size = static_cast<std::uint32_t>(sg_pool_edges_vec.size());
    pool.sg_pool_edges = sycl::malloc_device<std::uint32_t>(pool.sg_pool_edges_size, queue_);
    pool.sg_pool_vertices_size = static_cast<std::uint32_t>(sg_pool_vertices_vec.size());
    pool.sg_pool_vertices = sycl::malloc_device<std::uint32_t>(pool.sg_pool_vertices_size, queue_);
    pool.wi_pool_edges_size = static_cast<std::uint32_t>(wi_pool_edges_vec.size());
    pool.wi_pool_edges = sycl::malloc_device<std::uint32_t>(pool.wi_pool_edges_size, queue_);
    pool.wi_pool_vertices_size = static_cast<std::uint32_t>(wi_pool_vertices_vec.size());
    pool.wi_pool_vertices = sycl::malloc_device<std::uint32_t>(pool.wi_pool_vertices_size, queue_);

    queue_.copy(wg_pool_edges_vec.data(), pool.wg_pool_edges, pool.wg_pool_edges_size);
    queue_.copy(wg_pool_vertices_vec.data(), pool.wg_pool_vertices, pool.wg_pool_vertices_size);
    queue_.copy(sg_pool_edges_vec.data(), pool.sg_pool_edges, pool.sg_pool_edges_size);
    queue_.copy(sg_pool_vertices_vec.data(), pool.sg_pool_vertices, pool.sg_pool_vertices_size);
    queue_.copy(wi_pool_edges_vec.data(), pool.wi_pool_edges, pool.wi_pool_edges_size);
    queue_.copy(wi_pool_vertices_vec.data(), pool.wi_pool_vertices, pool.wi_pool_vertices_size);
    queue_.wait();


    // sycl::free(pool.tmp_buffer, queue_);
    // pool.tmp_buffer = nullptr;

    return pool;
}