"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 2: Programming assignment
Problem 2
"""
import math

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from torch import nn
from torch.autograd.functional import jacobian

torch.manual_seed(0)


class GCN(nn.Module):
    """
        Graph convolutional layer
    """

    def __init__(self, in_features, out_features, A_norm):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A_norm = A_norm
        self.weight = torch.nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, H):
        HW = torch.mm(H.float(), self.weight)
        AHW = torch.mm(self.A_norm.float(), HW)
        return F.relu(AHW)


"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200  # total number of nodes
i, j = 17, 27  # node ID
k = 0  # k-hop
G = nx.barabasi_albert_graph(n_V, 2, seed=seed)  # 2 is the number of edges to attach from a new node to existing nodes

# -- plot graph
layout = nx.spring_layout(G, seed=seed, iterations=400)  # build layout

"""
    Plot the effective range
"""
# -- plot neighborhood
# Compute the shortest path lengths from node 17 to reachable nodes, k Depth to stop the search
for source, color in zip([17, 27], ['red', 'magenta']):
    for hop in [2, 4, 6]:
        nodes = nx.single_source_shortest_path_length(G, source, cutoff=hop)
        nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
        im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color=color, node_size=100)
        plt.colorbar(im2)
        plt.show()
        plt.close()

# -- adj
A = nx.adjacency_matrix(G).toarray()
A_hat_np = A + np.identity(n=A.shape[0])
D_hat_np = np.squeeze(np.sum(np.array(A_hat_np), axis=1))
D_hat_inv_sqrt_np = np.diag(np.power(D_hat_np, -1 / 2))
A_norm = torch.from_numpy(np.dot(np.dot(D_hat_inv_sqrt_np, A_hat_np), D_hat_inv_sqrt_np))
A_norm = torch.tensor(A_norm)

# -- Initialize node attributes with one-hot vectors of node indices
index = np.array(list(G.nodes))
H = np.zeros((index.size, index.max() + 1), dtype=float)
H[np.arange(index.size), index] = 1
H = torch.tensor(H)

"""
    Influence score
"""


# -- Initialize the model

def MyModel1(H):
    gcn = GCN(200, 100, A_norm)
    x = gcn(H)
    return x


def MyModel2(H):
    gcn1 = GCN(200, 100, A_norm)
    gcn2 = GCN(100, 50, A_norm)
    gcn3 = GCN(50, 20, A_norm)
    x = gcn3(gcn2(gcn1(H)))
    return x


def MyModel3(H):
    gcn1 = GCN(200, 100, A_norm)
    gcn2 = GCN(100, 50, A_norm)
    gcn3 = GCN(50, 20, A_norm)
    gcn4 = GCN(20, 20, A_norm)
    gcn5 = GCN(20, 20, A_norm)
    x = gcn5(gcn4(gcn3(gcn2(gcn1(H)))))
    return x


# -- Influence score

inf_score = []
for source in [i, j]:
    for model in [MyModel1, MyModel2, MyModel3]:
        score = np.sum(np.sum(jacobian(model, H)[source].numpy(), axis=0), axis=1)
        inf_score.append(score)

for score, i in zip(inf_score[:3], [1, 2, 3]):
    print('Influence score on nodes vi=17 for Model {}'.format(i))
    print(score)
for score, i in zip(inf_score[3:], [1, 2, 3]):
    print('Influence score on nodes vi=27 for Model {}'.format(i))
    print(score)

# -- plot influence scores

for score in inf_score:
    nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
    im2 = nx.draw_networkx_nodes(G, pos=layout, node_color=score, node_size=100)
    plt.colorbar(im2)
    plt.show()
    plt.close()
