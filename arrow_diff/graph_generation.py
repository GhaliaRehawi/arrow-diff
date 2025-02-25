import logging
import torch
import torch.nn.functional as F

from time import time
from torch.nn import Module
from torch_geometric.utils import coalesce, degree, to_undirected
from typing import Optional, TypeVar

from arrow_diff.random_walks import generate_random_walks, random_walks_to_edge_index


# Create TypeVars for FloatTensor and LongTensor
FloatTensor = TypeVar('FloatTensor', torch.FloatTensor, torch.cuda.FloatTensor)
LongTensor = TypeVar('LongTensor', torch.LongTensor, torch.cuda.LongTensor)


@torch.no_grad()
def generate_graph(diffusion_model: Module, batch_size: int, walk_length: int, gnn: Module,
                   x: FloatTensor, deg_gt: FloatTensor, num_steps: int, num_walks_per_node: int = 1,
                   device: Optional[torch.device] = None, seed: Optional[int] = None,
                   verbose: bool = False) -> LongTensor:
    """
    Iteratively generates graphs in num_steps steps using degree guidance to sample start nodes and
    random walk diffusion to propose new edges.

    Args:
        diffusion_model: Module
            Trained Diffusion Model.
        batch_size: int
            Batch size that was used to train the Diffusion Model.
        walk_length: int
            The random walk length that was used to train the Diffusion Model.
        gnn: Module
            Trained GNN for edge classification.
        x: FloatTensor, shape: (num_nodes, num_node_features)
            Node features of the nodes in the original graph.
        deg_gt: FloatTensor, shape: (num_nodes,)
            FloatTensor containing the node degrees for every node in the original graph.
        num_steps: int
            Number of steps that are used to generate a graph.
        num_walks_per_node: int (optional, default: 1)
            Number of random walks to generate per start node.
        device: torch.device (optional, default: None)
            Device to perform computation on.
        seed: int (optional, default: None)
            Seed for reproducible sampling.
        verbose: bool (optional, default: False)
            Whether to use logging.info() to output information.

    Returns:
        edge_index: LongTensor, shape: (2, num_edges)
            Index tensor representing the edges of the generated graph.
    """
    # Set the gnn into evaluation mode
    gnn.eval()

    # Move the gnn parameters to the specified device
    gnn.to(device)

    # Move the node features x and deg_gt to the specified device
    x = x.to(device)
    deg_gt = deg_gt.to(device)

    num_nodes = deg_gt.size(0)

    # Initialize a generator
    generator = None if seed is None else torch.Generator(device=device)

    # Initialize an empty set of edges
    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    start_nodes = None

    for step in range(1, num_steps + 1):
        if verbose:
            logging.info(f'\tStep: {step}/{num_steps}')

        prev_edge_index_size = edge_index.size(1)

        if verbose:
            start_time = time()

        # Generate random walks from the start nodes
        random_walks = generate_random_walks(diffusion_model, walk_length, batch_size, num_nodes,
                                             start_nodes=start_nodes, num_walks_per_node=num_walks_per_node,
                                             device=device, seed=step * seed)

        if verbose:
            logging.info(f'\t\tGenerating {random_walks.size(0)} random walks took {time() - start_time:.3f} seconds')

        # Compute the new potential edges from the generated random walks
        edge_index_new = random_walks_to_edge_index(random_walks, num_nodes).to(device)

        if verbose:
            logging.info(f'\t\tAdding {edge_index_new.size(1)} potential edges to the graph')

        # Update edge_index by adding the new potential edges to edge_index
        edge_index = torch.cat([edge_index, edge_index_new], dim=1)

        # Sort edge_index and remove duplicated edges
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

        # Run the gnn on the graph with the original node features and the valid and potential new edges
        node_embeddings = gnn(x, edge_index)

        # Compute a probability for each potential new directed (because probs would be symmetric) edge how
        # likely it is valid based on the node embeddings
        row, col = edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        probs = torch.sigmoid(torch.sum(node_embeddings[row] * node_embeddings[col], dim=1))

        # Sample valid edges based on the computed probabilities
        if generator is not None:
            # Seed the generator for reproducible sampling
            generator.manual_seed(step * seed)

        # Sample valid edges and obtain their indices in edge_index
        idxs = torch.where(torch.bernoulli(probs, generator=generator))[0]

        # Update edge_index by only taking the valid edges
        edge_index = torch.stack([row[idxs], col[idxs]], dim=0)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        if verbose:
            logging.info(f'\t\tAdding {(edge_index.size(1) // 2) - prev_edge_index_size} valid edges to the graph')
            logging.info(f'\t\tThe current graph contains {edge_index.size(1) // 2} edges')

        # Compute the node degrees based on the valid edges
        deg = degree(edge_index[0], num_nodes)

        # Compute a probability for each node how likely it is a start node
        # (which needs more edges w.r.t. the original graph)
        probs = F.relu(deg_gt - deg)

        if torch.all(probs == 0):
            # Break if all probabilities are zero
            break

        # Normalize the probabilities
        probs /= probs.max()

        if generator is not None:
            # Seed the generator for reproducible sampling
            generator.manual_seed(step * seed)

        # Sample start nodes based on the computed probabilities and obtain their indices
        start_nodes = torch.where(torch.bernoulli(probs, generator=generator).bool())[0]

    if verbose:
        logging.info(f'Generated graph contains {edge_index.size(1) // 2} edges')

    return edge_index
