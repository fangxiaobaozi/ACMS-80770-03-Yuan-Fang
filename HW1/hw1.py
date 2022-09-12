"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 1: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy

# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)


# -- compute jaccard's similarity
# -- write my own function to compute Jaccard's similarity
def jaccard(node1, node2, G):
    neighbour1 = list(G.adj[node1])
    neighbour2 = list(G.adj[node2])
    num_inter = len(set(neighbour1) & set(neighbour2))
    num_union = len(set(neighbour1).union(set(neighbour2)))
    return round(num_inter / num_union, 2)


# -- get the similarity matrix
node_list = list(G.nodes)
num_node = len(node_list)
jaccard_matrix = np.zeros([num_node, num_node], dtype=float)
for i in range(num_node):
    for j in range(num_node):
        jaccard_matrix[i, j] = jaccard(node_list[i], node_list[j], G)

# -- print the similarity matrix
print(jaccard_matrix)

# -- function in networkx to compute Jaccard's similarity
pred = nx.jaccard_coefficient(G)
#
# -- add new edges representing similarities.
new_edges, metric = [], []
for u, v, p in pred:
    # G.add_edge(u, v)
    # print(f"({u}, {v}) -> {p:.8f}")
    new_edges.append((u, v))
    metric.append(p)

# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=G.edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity of "Ginori"
Ginori_edge, Ginori_metric = [], []
for item in new_edges:
    if 'Ginori' in item:
        Ginori_edge.append(item)
        Ginori_metric.append(metric[new_edges.index(item)])

ne = nx.draw_networkx_edges(G, edgelist=Ginori_edge, pos=layout, edge_color=np.asarray(Ginori_metric), width=4,
                            alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()
