import torch
import torch.nn.functional as F

from torch.nn import Module
from torch_geometric.utils import to_undirected
from typing import Optional, TypeVar


# Create TypeVar for LongTensor
LongTensor = TypeVar('LongTensor', torch.LongTensor, torch.cuda.LongTensor)


def sample_random_walks(rowptr: LongTensor, col: LongTensor, num_walks: int, walk_length: int, p: float = 1.0,
                        q: float = 1.0, node_idxs: Optional[LongTensor] = None,
                        generator: Optional[torch.Generator] = None, seed: Optional[int] = None) -> LongTensor:
    """
    Samples random walks from a graph.

    Args:
        rowptr: torch.LongTensor
            Pointer to the rows of the adjacency matrix with non-zero entries.
        col: torch.LongTensor
            Pointer to the columns of the adjacency matrix with non-zero entries.
        num_walks: int
            Number of start nodes to sample for the random walks.
        walk_length: int
            Length of the random walks.
        p: float (optional, default: 1.0)
            Control of revisiting a node.
        q: float (optional, default: 1.0)
            Control between breadth-first search and depth-first search.
        node_idxs: torch.LongTensor (optional, default: None), shape: (rowptr.size(0) - 1,)
            Original node indices of the subgraph.
        generator: torch.Generator (optional, default: None)
            Generator used for reproducible sampling of the start nodes for the random walks.
        seed: int (optional, default: None)
            Seed for reproducible sampling of the random walks.

    Returns:
        rw_nodes: torch.LongTensor, shape: (num_walks, walk_length)
            A torch.LongTensor containing the sampled random walks represented by their node IDs.
    """
    # Sample a batch of start nodes for the random walks
    start = torch.multinomial(torch.ones(rowptr.size(0) - 1, device=rowptr.device), num_walks, generator=generator)

    if seed is not None:
        # Set a seed for reproducible sampling of the random walks
        torch.manual_seed(seed)

    # Perform random walks from the start nodes on the graph
    rw_nodes, _ = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, p, q)

    if node_idxs is not None:
        # Map the indices in rw_nodes to the original node indices node_idxs
        rw_nodes = node_idxs[rw_nodes]

    return rw_nodes


@torch.no_grad()
def generate_random_walks(model: Module, walk_length: int, batch_size: int, num_nodes: int,
                          start_nodes: Optional[LongTensor] = None, num_walks_per_node: int = 1,
                          device: Optional[torch.device] = None, seed: Optional[int] = None) -> LongTensor:
    """
    Generates random walks from sequences containing all mask states except the first position, which is set to specific
    node IDs, using the trained model.

    Args:
        model: torch.nn.Module
            Model that predicts the probabilities for the reverse diffusion process.
        walk_length: int
            Walk length of the random walks to generate which equals the number of diffusion time steps.
        batch_size: int
            The batch size used for model training.
        num_nodes: int
            Number of nodes in the graph.
        start_nodes: LongTensor, shape: (num_start_nodes,) (optional, default: None)
            Subset of nodes to use as start nodes to generate random walks.
            If set to None, all num_nodes nodes will be used as start nodes.
        num_walks_per_node: int (optional, default: 1)
            Number of random walks to generate per start node.
        device: torch.device (optional, default: None)
            Device to perform computation on.
        seed: int (optional, default: None)
            Seed for reproducible generation of random walks.

    Returns:
        random_walks: torch.LongTensor, shape: (num_walks, walk_length)
            The generated num_walks random walks in index representation.
    """
    # Initialize the generator if a seed is set
    generator = None if seed is None else torch.Generator(device=device)

    # Set the model into evaluation mode
    model.eval()

    # Move the model parameters to the specified device
    model.to(device)

    # Create a LongTensor of all ones
    ones = torch.ones(batch_size, dtype=torch.long, device=device)

    if start_nodes is None:
        # If no start nodes are provided, consider all nodes in the graph as start nodes.
        start_nodes = torch.arange(num_nodes, dtype=torch.long, device=device)
    elif start_nodes.device != device:
        # Make sure that start_nodes is on the correct device
        start_nodes = start_nodes.to(device)

    if num_walks_per_node > 1:
        # If multiple random walks per node should be generated, repeat start_nodes num_walks_per_node times
        start_nodes = torch.repeat_interleave(start_nodes, num_walks_per_node)

    # Specify how many random walks need to be generated in total
    num_walks = start_nodes.size(0)

    random_walks = []

    # Compute batch_size random walks per iteration
    for batch in range(1, num_walks // batch_size + int(bool(num_walks % batch_size)) + 1):
        # Generate one permutation per random walk
        permutations = []

        for i in range(1, batch_size + 1):
            if generator is not None:
                # Seed the generator for reproducibility
                generator.manual_seed(batch * i * seed)

            # Generate a permutation of size walk_length - 1
            permutations.append(torch.randperm(walk_length - 1, generator=generator, device=device))

        # Stack the permutations along the batch dimension to a single LongTensor
        perm = torch.stack(permutations, dim=0)

        # Append index 0 to all permutation indices, which are increased by 1
        perm = torch.cat([torch.zeros((batch_size, 1), dtype=torch.long, device=device), perm + 1], dim=1)

        # Take the current batch of start nodes
        start_nodes_batch = start_nodes[(batch - 1) * batch_size:min(batch * batch_size, num_walks)]

        if start_nodes_batch.size(0) < batch_size:
            # Pad start_nodes_batch with zeros
            start_nodes_batch = F.pad(start_nodes_batch, (0, batch_size - start_nodes_batch.size(0)))

        # Create a LongTensor with all states set to mask (= num_nodes)
        rw_nodes = torch.full((batch_size, walk_length - 1), num_nodes, device=device)

        # Append the start nodes to rw_nodes at the first position
        rw_nodes = torch.cat([start_nodes_batch.unsqueeze(1), rw_nodes], dim=1)

        # Iterate over the walk length (= diffusion time steps) and start at time_step 2 (= index 1)
        # because the start nodes at position 1 (= index 0) were manually set
        for time_step in range(1, walk_length):
            # Compute the indices for sigma(= t)
            idxs = torch.where(perm == time_step)

            # Get the model predictions (log-probabilities, shape: (batch_size, walk_length, num_nodes))
            # for sigma(= t), for each element in the batch, shape: (batch_size, num_nodes)
            log_prob = model(rw_nodes, time_step * ones)[idxs]

            if generator is not None:
                # Seed the generator for reproducibility
                generator.manual_seed(batch * time_step * seed)

            # Sample the node indices for the current time step: x_k ~ p_\theta(x_k | x_{sigma(< t)}); k \in sigma(= t)
            samples = torch.multinomial(torch.exp(log_prob), 1, generator=generator).reshape(batch_size)

            # Assign the sampled node indices to rw_nodes at position sigma(= t) of the current time step
            rw_nodes[idxs] = samples

        # Add the batch of sampled random walks to random_walks
        random_walks.append(rw_nodes.cpu())

    # Concatenate the generated random walks from all batches and only take num_walks random walks
    random_walks = torch.cat(random_walks, dim=0)[:num_walks]

    return random_walks


def random_walks_to_edge_index(random_walks: LongTensor, num_nodes: int) -> LongTensor:
    """
    Compute the edge index tensor for the edges inside the provided random walks.

    Args:
        random_walks: LongTensor, shape: (num_walks, walk_length)
            The input random walks.
        num_nodes: int
            The number of nodes of the input graph.

    Returns:
        edge_index: LongTensor, shape: (2, num_edges)
            LongTensor representing the edges in random_walks.
    """
    # Get every node in rows and the node it connects to in cols
    row = random_walks[:, :-1].reshape(-1)
    col = random_walks[:, 1:].reshape(-1)

    edge_index = torch.stack([row, col], dim=0)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return edge_index
