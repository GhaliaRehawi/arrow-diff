import torch
import networkx as nx
import numpy as np
import powerlaw

from torch import LongTensor
from typing import List, Tuple, Union


def edge_overlap(edge_index_1: LongTensor, edge_index_2: LongTensor,
                 return_edges: bool = False) -> Union[int, Tuple[int, LongTensor]]:
    """
    Compares two sets of edges and returns the intersection of both.

    Args:
        edge_index_1: torch.LongTensor, shape: (2, num_edges_1)
            Indices representing the edges of the graph.
        edge_index_2: torch.LongTensor, shape: (2, num_edges_2)
            Indices representing the edges of the graph.
        return_edges: bool (optional, default: False)
            Whether to return the overlapping edges.

    Returns:
        num_edges: int
            Number of overlapping edges.
        edges: torch.LongTensor, shape: (2, num_edges)
            Indices with the intersection of the two sets of edge indices.
    """
    set_1 = set(zip(*edge_index_1.tolist()))
    set_2 = set(zip(*edge_index_2.tolist()))

    intersection = set_1.intersection(set_2)

    if return_edges:
        # Convert intersection set to a torch.LongTensor
        edges = torch.tensor(list(zip(*intersection)), dtype=torch.long).reshape(2, len(intersection))

        return len(intersection), edges
    else:
        return len(intersection)


def get_graph_statistics(graph: nx.Graph) -> Tuple[float, float, float, float, int, int, List[int]]:
    """
    Get graph statistics for a given nx.Graph.

    Args:
        graph: nx.Graph
            Graph for which to compute the statistics.

    Returns:
        degree_assort: float
            Degree assortativity of the graph.
        avg_clustering_coef: float
            Avg Clustering coefficient of the graph.
            It is the mean of local clustering coefficient: the fraction of links among neighbouring nodes of a node
            and all possible links between those nodes,
        global_clustering_coef: float
            Global Clustering coefficient of the graph.
            The global clustering coefficient is the number of closed triplets (or 3 x triangles)
            over the total number of triplets (both open and closed).
        power_law_exp: float
            The power law coefficient of the degree distribution of the input graph (similar to NetGAN)
        num_triangles: int
            Number of triangles in the graph.
        max_degree: int
            Maximum node degree in the graph.
        connected_component_sizes: List[int]
            Sizes of the connected components in the graph sorted in descending order.
    """

    # Calculate degree assortativity
    degree_assortativity = nx.degree_pearson_correlation_coefficient(graph)

    # Calculate the average clustering coefficient of the graph
    avg_clustering_coef = nx.average_clustering(graph)

    # Calculate the global clustering coefficient: is the number of closed triplets (or 3 x triangles)
    # over the total number of triplets (both open and closed) = 3 * num_triangles / num_claws
    degrees = [item[1] for item in list(graph.degree)]
    num_claws = np.sum(np.fromiter((1 / 6. * x * (x - 1) * (x - 2) for x in degrees), dtype=float))

    # Calculate the number of triangles in the graph
    # When computing triangles for the entire graph each triangle is counted three times, once at each node
    num_triangles = sum(nx.triangles(graph).values()) // 3

    # The global clustering coefficient
    global_clustering_coef = 3 * num_triangles / num_claws

    # Calculate max degree of the graph
    max_degree = max(graph.degree, key=lambda x: x[1])[1]

    # Calculate the power law exponent
    power_law_exp = powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha

    connected_component_sizes = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]

    return (degree_assortativity, avg_clustering_coef, global_clustering_coef, power_law_exp, num_triangles, max_degree,
            connected_component_sizes)
