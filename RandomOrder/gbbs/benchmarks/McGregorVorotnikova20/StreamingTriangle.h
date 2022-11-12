#include <unordered_set>
#include <stack>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdlib>

#include "gbbs/gbbs.h"
#include "gbbs/dynamic_graph_io.h"
#include "benchmarks/TriangleCounting/ShunTangwongsan15/Triangle.h"
#include "gbbs/helpers/sparse_table.h"
#include "sparse_set.h"

namespace gbbs {
using edge_type = std::pair<uintE, uintE>;
using random = parlay::random;

auto f = [&] (uintE u, uintE v, uintE w) { };

double mean(const std::vector<double> &v)
{
    double sum = 0;

    for (auto &each: v)
        sum += each;

    return sum / v.size();
}

double sd(const std::vector<double> &v)
{
    double square_sum_of_difference = 0;
    double mean_var = mean(v);
    auto len = v.size();

    double tmp;
    for (auto &each: v) {
        tmp = each - mean_var;
        square_sum_of_difference += tmp * tmp;
    }

    return std::sqrt(square_sum_of_difference / (len - 1));
}

std::pair<size_t, size_t> _binary_pair_search(sequence<edge_type> I, uintE value, size_t n) {
    size_t start = 0;
    size_t end = n;
    size_t last_smallest = 0;
    bool found_smallest = false;

    size_t smallest_size = 16;

    while (end - start > smallest_size) {
        size_t mid = start + (end - start) / 2;
        if (I[mid].first >= value)
            end = mid;
        else {
            start = mid + 1;
            last_smallest = mid;
            if (start < n && I[start].first >= value) {
                found_smallest = true;
                break;
            }
        }
    }

    if (!found_smallest) {
        for (size_t i = start; i < end; i++) {
            if (I[i].first < value)
                last_smallest = i;
            else
                break;
        }
    }

    start = 0;
    end = n;
    size_t last_largest = end;
    bool found_largest = false;

    while (end - start > smallest_size) {
        size_t mid = start + (end - start) / 2;
        if (I[mid].first <= value) {
            start = mid + 1;
            if (start < n && I[start].first > value) {
                found_largest = true;
                last_largest = start;
                break;
            }
        } else {
            end = mid;
            last_largest = end;
            if (end - 1 >= 0 && I[end-1].first <= value) {
                found_largest = true;
                break;
            }
        }
    }

    if (!found_largest) {
        for (size_t i = start; i < end; i++) {
            if (I[i].first > value)
                last_largest = i;
                break;
        }
    }

    return std::make_pair(last_smallest, last_largest);
}

template <class W>
inline void ApproximateTriangles (BatchDynamicEdges<W>& batch_edge_list,
        size_t trials,
        commandLine& P,
        double eps) {
    auto& batch = batch_edge_list.edges;

    uintE num_edges = batch.size();

    std::vector<double> errors;
    auto rand = random();
    auto whole_graph = dynamic_edge_list_to_symmetric_graph(
        batch, batch.size());
    auto counts = Triangle(whole_graph, f, "kcore", P);
    size_t triangle_real_count = counts.first;

    auto vertices = sequence<uintE>(2 * batch.size(), 0);
    parallel_for(0, batch.size(), [&] (size_t i){
        vertices[2 * i] = batch[i].from;
        vertices[2 * i + 1] = batch[i].to;
    });

    parlay::sample_sort_inplace(make_slice(vertices), std::less<uintE>());
    auto bool_seq = parlay::delayed_seq<bool>(vertices.size() + 1, [&] (size_t i) {
        return (i == 0) || (i == vertices.size())
            || (vertices[i-1] != vertices[i]);
    });

    auto starts = parlay::pack_index(bool_seq);
    auto unique_vertices = sequence<uintE>(starts.size() - 1, 0);
    parallel_for(0, starts.size() - 1, [&] (size_t i) {
        unique_vertices[i] = vertices[starts[i]];
    });

    for (size_t trial = 0; trial < trials; trial++) {
        std::srand(unsigned(std::time(0)));
        std::random_shuffle(batch.begin(), batch.end());
        uintE num_triangles = 0;

        double ind = floor(log2(sqrt(triangle_real_count)));
        double p = std::min(1.0, (10 * log2(unique_vertices.size())) / (pow(2, ind) * pow(eps, 2)));
        double r = 1/(sqrt(triangle_real_count) * eps);
        size_t S = floor(r * num_edges);
        size_t q = ceil((pow(2, ind)/sqrt(triangle_real_count)) * num_edges);
        auto cutoff = ceil(p * sqrt(triangle_real_count));

        size_t approx_count = 0;
        auto space_used = 0;
        auto f = [&] (uintE u, uintE v, uintE w) { };

        if (S < batch.size()) {

            auto sampled_vertices = sequence<uintE>(unique_vertices.size(), UINT_MAX);
            auto num_verts_to_sample = ceil(p * unique_vertices.size());

            parallel_for(0, unique_vertices.size(), [&] (size_t i){
                auto v = unique_vertices[i];
                auto r_v = rand.fork(v);
                auto random_num = r_v.rand() % unique_vertices.size();

                if (random_num <= num_verts_to_sample)
                    sampled_vertices[i] = unique_vertices[i];
            });
            rand = rand.next();

            sampled_vertices = parlay::filter(sampled_vertices, [&](uintE vertex) {
                return vertex != UINT_MAX;
            });

            space_used += S;

            auto sampled_vertices_set = sparse_set<uintE>();
            sampled_vertices_set.resize(sampled_vertices.size());
            parallel_for(0, sampled_vertices.size(), [&] (size_t i){
                sampled_vertices_set.insert(sampled_vertices[i]);
            });

            auto O_indices = sequence<uintE>(q, UINT_MAX);

            parallel_for(0, q, [&] (size_t i){
                auto edge = batch[i];
                uintE u = batch[i].from;
                uintE v = batch[i].to;
                if (sampled_vertices_set.contains(u) || sampled_vertices_set.contains(v)) {
                    O_indices[i] = i;

                    if (i > S) {
                        space_used += 1;
                    }
                }
            });

            O_indices = parlay::filter(O_indices, [&](size_t o_index){
                return o_index != UINT_MAX;
            });

            auto O_verts = sparse_set<uintE>();
            O_verts.resize(2 * O_indices.size());

            auto O_edges = std::vector<DynamicEdge<W>>(O_indices.size());
            parallel_for(0, O_edges.size(), [&] (size_t i){
                O_edges[i] = batch[O_indices[i]];
                O_verts.insert(O_edges[i].from);
                O_verts.insert(O_edges[i].to);
            });

            auto O_graph = dynamic_edge_list_to_symmetric_graph(O_edges, O_edges.size());

            auto light_S_indices = sequence<uintE>(S, UINT_MAX);
            auto light_S_verts = sparse_set<uintE>();
            light_S_verts.resize(2 * light_S_indices.size());

            parallel_for(0, S, [&] (size_t i) {
                uintE u = batch[i].from;
                uintE v = batch[i].to;
                auto our_neighbors = O_graph.get_vertex(u).out_neighbors();

                auto their_neighbors = O_graph.get_vertex(v).out_neighbors();
                auto o_count = our_neighbors.intersect_f_par(&their_neighbors, f);

                if (o_count < cutoff) {
                    light_S_indices[i] = i;
                    light_S_verts.insert(u);
                    light_S_verts.insert(v);
                }
            });

            light_S_indices = parlay::filter(light_S_indices, [&] (size_t i) {
                return i != UINT_MAX;
            });

            auto light_S = std::vector<DynamicEdge<W>>(light_S_indices.size());
            parallel_for(0, light_S_indices.size(), [&] (size_t i){
                light_S[i] = batch[light_S_indices[i]];
            });

            auto S_light_graph = dynamic_edge_list_to_symmetric_graph(light_S, light_S.size());
            auto pair_count = Triangle(S_light_graph, f, "kcore", P);
            uintE num_base_triangles = pair_count.first;

            num_triangles += num_base_triangles / pow(r, 2);
            auto end_index = S;

            auto O_light_indices = sequence<uintE>(O_edges.size(), UINT_MAX);
            auto O_heavy_indices = sequence<uintE>(O_edges.size(), UINT_MAX);

            auto O_light_verts = sparse_set<uintE>();
            auto O_heavy_verts = sparse_set<uintE>();
            O_light_verts.resize(2 * O_light_indices.size());
            O_heavy_verts.resize(2 * O_heavy_indices.size());

            parallel_for(0, O_edges.size(), [&] (size_t i){
                uintE u = O_edges[i].from;
                uintE v = O_edges[i].to;

                auto our_neighbors = O_graph.get_vertex(u).out_neighbors();
                auto their_neighbors = O_graph.get_vertex(v).out_neighbors();

                auto o_count = our_neighbors.intersect_f_par(&their_neighbors, f);

                if (o_count <= cutoff) {
                    O_light_indices[i] = i;
                    O_light_verts.insert(u);
                    O_light_verts.insert(v);
                } else {
                    O_heavy_indices[i] = i;
                    O_heavy_verts.insert(u);
                    O_heavy_verts.insert(v);
                }
            });
            O_light_indices = parlay::filter(O_light_indices, [&] (size_t index) {
                return index != UINT_MAX;
            });

            O_heavy_indices = parlay::filter(O_heavy_indices, [&] (size_t index) {
                return index != UINT_MAX;
            });

            auto O_light = std::vector<DynamicEdge<W>>(O_light_indices.size());
            auto O_heavy = std::vector<DynamicEdge<W>>(O_heavy_indices.size());

            parallel_for(0, O_light_indices.size(), [&] (size_t i){
                O_light[i] = O_edges[O_light_indices[i]];
            });

            parallel_for(0, O_heavy_indices.size(), [&] (size_t i){
                O_heavy[i] = O_edges[O_heavy_indices[i]];
            });

            auto O_light_graph = dynamic_edge_list_to_symmetric_graph(O_light, O_light.size());
            auto O_heavy_graph = dynamic_edge_list_to_symmetric_graph(O_heavy, O_heavy.size());

            for (size_t edge_ind = end_index; edge_ind < batch.size(); edge_ind++) {
                // Run static triangle counting on the graph with the additional edge
                uintE u = batch[edge_ind].from;
                uintE v = batch[edge_ind].to;
                auto o_count = 0;
                if (O_verts.contains(u) && O_verts.contains(v)) {
                    auto our_neighbors = O_graph.get_vertex(u).out_neighbors();

                    auto their_neighbors = O_graph.get_vertex(v).out_neighbors();
                    o_count = our_neighbors.intersect_f_par(&their_neighbors, f);
                }

                if (o_count <= cutoff) {
                    auto s_count = 0;

                    if (light_S_verts.contains(u) && light_S_verts.contains(v)) {
                        auto s_our_neighbors = S_light_graph.get_vertex(u).out_neighbors();

                        auto s_their_neighbors = S_light_graph.get_vertex(v).out_neighbors();
                        s_count = s_our_neighbors.intersect_f_par(&s_their_neighbors, f);
                    }

                    num_triangles += 1.0 * s_count / (3 * pow(r, 2));
                } else if (edge_ind > q){
                    auto t_0 = 0;
                    auto t_1 = 0;
                    auto t_2 = 0;

                    if (O_heavy_verts.contains(u) && O_light_verts.contains(v)) {
                        auto our_heavy_n = O_heavy_graph.get_vertex(u).out_neighbors();
                        auto their_light_n = O_light_graph.get_vertex(v).out_neighbors();
                        t_1 += our_heavy_n.intersect_f_par(&their_light_n, f);
                    }

                    if (O_light_verts.contains(u) && O_heavy_verts.contains(v)) {
                        auto our_light_n = O_light_graph.get_vertex(u).out_neighbors();
                        auto their_heavy_n = O_heavy_graph.get_vertex(v).out_neighbors();
                        t_1 += our_light_n.intersect_f_par(&their_heavy_n, f);
                    }

                    if (O_light_verts.contains(u) && O_light_verts.contains(v)) {
                        auto our_light_n = O_light_graph.get_vertex(u).out_neighbors();
                        auto their_light_n = O_light_graph.get_vertex(v).out_neighbors();
                        t_0 += our_light_n.intersect_f_par(&their_light_n, f);
                    }

                    if (O_heavy_verts.contains(u) && O_heavy_verts.contains(v)) {
                        auto our_heavy_n = O_heavy_graph.get_vertex(u).out_neighbors();
                        auto their_heavy_n = O_heavy_graph.get_vertex(v).out_neighbors();
                        t_2 += our_heavy_n.intersect_f_par(&their_heavy_n, f);
                    }

                    num_triangles += (1.0 / p) * (t_0 + float(t_1)/2 + float(t_2) / 3);
                }

            }

            approx_count = num_triangles;
        } else
            approx_count = triangle_real_count;

        auto error = std::abs(1 - float(approx_count) / triangle_real_count);
        errors.push_back(error);
        std::cout << "### Trial #: " << trial << std::endl;
        std::cout << "### S: " << space_used << std::endl;
        std::cout << "### m: " << num_edges << std::endl;
        std::cout << "### Exact Triangle Count: " << triangle_real_count << std::endl;
        std::cout << "### Approx Count: " << approx_count << std::endl;
        std::cout << "### Error: " << error << std::endl;
    }

    auto m_val = mean(errors);
    auto sd_val = sd(errors);
    std::cout << "### Mean: " << m_val << std::endl;
    std::cout << "### Standard Div: " << sd_val << std::endl;
}

}  // namespace gbbs
