import math

import numpy as np
import random
from copy import deepcopy

def get_adjacency_lists(adj_dict, edges):
    adj_lists = deepcopy(adj_dict)

    for edge in edges:
        if edge[0] not in adj_lists:
            adj_lists[edge[0]] = [edge[1]]
        else:
            adj_lists[edge[0]].append(edge[1])
        if edge[1] not in adj_lists:
            adj_lists[edge[1]] = [edge[0]]
        else:
            adj_lists[edge[1]].append(edge[0])
    for node in adj_lists:
        adj_lists[node].sort()
    return adj_lists

def find_num_intersects(edges1, edges2):
    intersection = set(edges1).intersection(edges2)
    return len(intersection)

def find_num_node_intersects(edge_dict, neighbor_list):
    count = 0
    for i in range(len(neighbor_list)):
        a = neighbor_list[i]
        for j in range(i + 1, len(neighbor_list)):
            b = neighbor_list[j]
            if (a, b) in edge_dict or (b, a) in edge_dict:
                count+=1
    return count

def get_node_buckets_and_counts(edges, error, start, end,
        step_size, bucket_1, bucket_2, whole_graph = None):
    bucket_1_counts = {}
    edge_dict = {}
    for edge in edges[start: start + step_size]:
        edge_dict[edge] = 1

    for key in bucket_1:
        intersects = find_num_node_intersects(edge_dict, bucket_1[key])
        if intersects > 0:
            bucket = math.ceil(math.log(intersects, error))
            if bucket not in bucket_1_counts:
                bucket_1_counts[bucket] = 1
            else:
                bucket_1_counts[bucket] += 1

    for edge in edges[start + step_size: start + 2 * step_size]:
        edge_dict[edge] = 1

    bucket_2_counts = {}
    for key in bucket_2:
        intersects = find_num_node_intersects(edge_dict, bucket_2[key])
        if intersects > 0:
            bucket = math.ceil(math.log(intersects, error))
            if bucket not in bucket_2_counts:
                bucket_2_counts[bucket] = 1
            else:
                bucket_2_counts[bucket] += 1

    count_whole_graph = 0
    if not whole_graph == None:
        for edge in edges[start + 2 * step_size: len(edges)]:
            edge_dict[edge] = 1

        for key in whole_graph:
            intersects = find_num_node_intersects(edge_dict, whole_graph[key])
            count_whole_graph += intersects

    b1 = []
    b2 = []
    for i in range(115):
        if i not in bucket_1_counts:
            b1.append(0)
        else:
            b1.append(bucket_1_counts[i])

        if i not in bucket_2_counts:
            b2.append(0)
        else:
            b2.append(bucket_2_counts[i])

    return ([b1, b2], count_whole_graph)

def get_buckets_and_counts(edges, error, start, end,
        step_size, bucket_1, bucket_2, whole_graph):
    bucket_1_counts = {}
    count_1 = 0
    for edge in edges[start: start + step_size]:
        intersects = find_num_intersects(bucket_1[edge[0]], bucket_1[edge[1]])
        if intersects > 0:
            bucket = math.ceil(math.log(intersects, error))
            if bucket not in bucket_1_counts:
                bucket_1_counts[bucket] = 1
            else:
                bucket_1_counts[bucket] += 1
        count_1 += intersects

    bucket_2_counts = {}
    count_2 = 0
    for edge in edges[start: start + 2 * step_size]:
        intersects = find_num_intersects(bucket_2[edge[0]], bucket_2[edge[1]])
        if intersects > 0:
            bucket = math.ceil(math.log(intersects, error))
            if bucket not in bucket_2_counts:
                bucket_2_counts[bucket] = 1
            else:
                bucket_2_counts[bucket] += 1
        count_2 += intersects

    whole_graph_counts = {}
    count_whole_graph = 0
    for edge in edges[start: end]:
        intersects = find_num_intersects(whole_graph[edge[0]], whole_graph[edge[1]])
        if intersects > 0:
            bucket = math.ceil(math.log(intersects, error))
            if bucket not in whole_graph_counts:
                whole_graph_counts[bucket] = 1
            else:
                whole_graph_counts[bucket] += 1
        count_whole_graph += intersects

    b1 = []
    b2 = []
    w = []

    for i in range(115):
        if i not in bucket_1_counts:
            b1.append(0)
        else:
            b1.append(bucket_1_counts[i])

        if i not in bucket_2_counts:
            b2.append(0)
        else:
            b2.append(bucket_2_counts[i])

        if i not in whole_graph_counts:
            w.append(0)
        else:
            w.append(whole_graph_counts[i])

    return ([b1, b2, w], [count_1, count_2, count_whole_graph])

def write_to_file(buckets, counts, bucket_dir, count_dir, data_file_name, start,
        end, step_size):
    buckets_file = open(bucket_dir + data_file_name, "a")

    for i in range(2):
        if i == 0:
            buckets_file.write(str(end - start) + " " + str(step_size) + " ")
        for bucket in buckets[i]:
            buckets_file.write(str(bucket) + " ")
    buckets_file.write(str(counts))
    buckets_file.write("\n")
    buckets_file.close()

def read_counts(counts_file_name):
    counts_file = open(counts_file_name, "r")
    counts = []
    for line in counts_file.readlines():
        count = line.split()[-1]
        counts.append(int(count))
    return counts

def load_dataset(dataset_file):
    dataset = open(dataset_file, "r")
    X = []
    y = []
    for line in dataset.readlines():
        cur_X = []
        buckets = line.split()
        for i in range(232):
            cur_X.append(int(buckets[i]))
        y.append(int(buckets[232]))
        X.append(cur_X)
    return np.array(X), np.array(y)

def permute(graph):
    random.shuffle(graph)

def sample_vertices(verts, p):
    sampled_verts = {}
    for v in verts:
        i = random.random()
        if i <= p:
            sampled_verts[v] = 1
    return sampled_verts

def get_vertices_and_adj_list(graph):
    adj_lists = {}
    verts = {}
    for edge in graph:
        u, v = edge
        verts[u] = 1
        verts[v] = 1

        if not u in adj_lists:
            adj_lists[u] = {}
            adj_lists[u][v] = 1
        else:
            adj_lists[u][v] = 1

        if not v in adj_lists:
            adj_lists[v] = {}
            adj_lists[v][u] = 1
        else:
            adj_lists[v][u] = 1
    return verts, adj_lists

def get_o(graph, q, sampled_vertices, space):
    prefix = graph[:int(q * len(graph))]
    O = {}
    count = 0
    for edge in prefix:
        u, v = edge
        if u in sampled_vertices or v in sampled_vertices:
            O[edge] = 1
        count += 1
    print("Space:", count)
    return O

def incident_triangles(edge, adj_list, O):
    u, v = edge
    inc_t = 0
    if u in adj_list and v in adj_list:
        adj_u = adj_list[u]
        adj_v = adj_list[v]

        for key in adj_u:
            if ((u, key) in O or (key, u) in O) and key in adj_v and ((v, key) in O or (key, v) in O):
                inc_t += 1
    return inc_t

def incident_triangles_by_weight(edge, adj_list, O, oracle, cutoff):
    u, v = edge
    adj_u = adj_list[u]
    adj_v = adj_list[v]
    t0, t1, t2 = 0

    for key in adj_u:
        t = 0
        if ((u, key) in O or (key, u) in O) and key in adj_v and ((v, key) in O or (key, v) in O):
            if (u, key) in O:
                if O[(u, key)] > cutoff:
                    t += 1
            elif O[(key, u)] > cutoff:
                t += 1
            if (v, key) in O:
                if O[(v, key)] > cutoff:
                    t += 1
            elif O[(key, v)] > cutoff:
                t += 1
            if t == 0:
                t0 += 1
            elif t == 1:
                t1 += 1
            else:
                t2 += 1
    return t0, t1, t2

def mv_count_triangles(graph, O, S, r, q, i, p, adj_list, T):
    count = 0
    oracle = {}
    S_light = {}
    cutoff = p * math.sqrt(T)
    for edge in S:
        inc_t = incident_triangles(edge, adj_list, O)
        if inc_t < cutoff:
            S_light[edge] = 1
        oracle[edge] = inc_t

    for edge in O:
        if edge not in oracle:
            inc_t = incident_triangles(edge, adj_list, O)
            oracle[edge] = inc_t

    P_h = {}
    for j in range(len(graph)):
        edge = graph[j]
        if edge in oracle and oracle[edge] < cutoff:
            inc_t = incident_triangles(edge, adj_list, S_light)
            count += 1.0/(3 * r ** 2) * inc_t
        else:
            category = incident_triangles(edge, adj_list, O)
            if category < cutoff:
                inc_t = incident_triangles(edge, adj_list, S_light)
                count += 1.0/(3 * r ** 2) * inc_t
            else:
                if j > q * len(graph):
                    t0, t1, t2 = incident_triangles_by_weight(edge, adj_list, O, oracle, cutoff)
                    if t0 + t1 + t2 >= cutoff:
                        count += (1/p) * (t0 + float(t1)/2 + float(t2)/3)
    return count

def merge_adj_lists(adj_1, adj_2):
    merged_adj_list = {}
    for key in adj_1:
        merged_adj_list[key] = adj_1[key]

    for key in adj_2:
        if key not in merged_adj_list:
            merged_adj_list[key] = adj_2[key]
        else:
            for k in adj_2[key]:
                merged_adj_list[key][k] = 1

    return merged_adj_list
