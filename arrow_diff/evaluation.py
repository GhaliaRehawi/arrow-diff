from torch import LongTensor
from torch_geometric.data import Data
from torch_geometric.utils import contains_self_loops, is_undirected, remove_self_loops, to_networkx, to_undirected
from typing import List, Tuple

from arrow_diff.metrics import edge_overlap, get_graph_statistics


def evaluate_predicted_graph(edge_index_pred: LongTensor, edge_index_gt: LongTensor, num_nodes: int) \
        -> Tuple[int, float, float, float, float, int, int, List[int]]:
    if contains_self_loops(edge_index_pred):
        edge_index_pred, _ = remove_self_loops(edge_index_pred)

    if not is_undirected(edge_index_pred, num_nodes=num_nodes):
        edge_index_pred = to_undirected(edge_index_pred, num_nodes=num_nodes)

    # Create a PyTorch Geometric Data object that represents the generated graph
    data_pred = Data(edge_index=edge_index_pred, num_nodes=num_nodes)

    # Convert data to a nx.Graph
    graph_pred = to_networkx(data_pred, to_undirected=True)

    # Compute the edge overlap of the generated graph with the original graph
    eo = edge_overlap(edge_index_pred, edge_index_gt) // 2

    # Compute graph statistics for the generated graph
    (degree_assort, avg_clustering_coef, global_clustering_coef, power_law_exp, num_triangles, max_degree,
     connected_component_sizes) = get_graph_statistics(graph_pred)

    return (eo, degree_assort, avg_clustering_coef, global_clustering_coef, power_law_exp, num_triangles, max_degree,
            connected_component_sizes)
