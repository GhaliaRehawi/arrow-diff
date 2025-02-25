import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
import warnings

from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from torch import LongTensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Planetoid
from torch_geometric.utils import is_undirected, subgraph, to_networkx, from_networkx
from typing import Optional, Tuple


def load_data(dataset_name: str, path: Optional[str] = None, seed: Optional[int] = None) -> Data:
    """
    Return a PyTorch Geometric Data object representing a single graph from a one-graph dataset.
    In case of 'Cora_ML', 'Cora', 'DBLP' and 'CiteSeer', the largest connected component is selected.

    Args:
        dataset_name: str
            Either 'Cora_ML', 'Cora', 'CiteSeer', 'DBLP', 'PubMed' or 'SBM'.
        path: str (optional, default: None)
            Path to the folder that contains the data for the 'Cora_ML', 'Cora', 'CiteSeer' 'DBLP', and 'PubMed'
            datasets.
        seed: int (optional, default: None)
            Seed for reproducible generation of the Stochastic Block Model (SBM) graph.

    Returns:
        data: torch_geometric.data.Data
            A PyTorch Geometric Data object representing a graph.
    """
    if dataset_name == 'SBM':
        sizes = [100, 60, 200]

        probs = [[0.2, 0.02, 0.005],
                 [0.02, 0.3, 0.01],
                 [0.005, 0.01, 0.1]]

        graph = nx.stochastic_block_model(sizes, probs, seed=seed, directed=False, selfloops=False)

        # Note: This is needed to make from_networkx() working for this graph
        graph.graph = dict()

        data = from_networkx(graph)
    else:
        # Initialize the dataset
        dataset = CitationFull(root=path, name=dataset_name)

        # Access the single graph inside the dataset
        data = dataset[0]

        if dataset_name in ['Cora_ML', 'Cora', 'CiteSeer', 'DBLP']:
            # Select the largest connected component of the original graph

            # Check if the graph is undirected
            undirected = is_undirected(data.edge_index, num_nodes=data.num_nodes)

            # Convert the graph to a nx.Graph or nx.DiGraph (if the graph is connected)
            graph = to_networkx(data, to_undirected=undirected)

            # Compute a generator holding sets with the node indices for all connected components
            connected_components = nx.connected_components(graph)

            # Compute the largest connected component
            largest_connected_component = torch.tensor(list(max(connected_components, key=len)))

            # Extract the induced subgraph by the nodes with the indices in largest_connected_component
            edge_index, _ = subgraph(largest_connected_component, data.edge_index, relabel_nodes=True,
                                     num_nodes=data.num_nodes)

            # Creat a new PyTorch Geometric Data object holding the edges of the largest connected component of the
            # original graph
            data = Data(x=data.x[largest_connected_component], edge_index=edge_index, y=data.y[largest_connected_component],
                        num_nodes=largest_connected_component.size(0))

    return data


# Code taken from https://github.com/danielzuegner/netgan/blob/master/netgan/utils.py
def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)

    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix

    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


# Code taken from https://github.com/danielzuegner/netgan/blob/master/netgan/utils.py
def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """
    Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False

    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges

    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = np.array(list(nx.maximal_matching(nx.DiGraph(A))))
                not_in_cover = np.array(list(set(range(N)).difference(hold_edges.flatten())))

                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                min_size = hold_edges.shape[0] + len(not_in_cover)
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))

                d_nic = d[not_in_cover]

                hold_edges_d1 = np.column_stack((not_in_cover[d_nic > 0],
                                                 np.row_stack(map(np.random.choice,
                                                                  A[not_in_cover[d_nic > 0]].tolil().rows))))

                if np.any(d_nic == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, not_in_cover[d_nic == 0]].T.tolil().rows)),
                                                     not_in_cover[d_nic == 0]))
                    hold_edges = np.row_stack((hold_edges, hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = np.row_stack((hold_edges, hold_edges_d1))

            else:
                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        test_zeros = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        test_zeros = np.row_stack(test_zeros)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def train_val_test_split_graph(data: Data, seed: Optional[int] = None) \
        -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor, LongTensor]:
    """
    Split a graph into training, validation and test edges, and validation and test non-edges.

    Args:
        data: torch_geometric.data.Data
            Data object representing a graph.
        seed: int (optional, default: None)
            Seed for reproducible splitting.

    Returns:
        train_edge_index: torch.LongTensor, shape: (2, num_train_edges)
            Training edges.
        val_edge_index: torch.LongTensor, shape: (2, num_val_edges)
            Validation edges.
        val_non_edge_index: torch.LongTensor, shape: (2, num_val_non_edges)
            Validation non-edges.
        test_edge_index: torch.LongTensor, shape: (2, num_test_edges)
            Test edges.
        test_non_edge_index: torch.LongTensor, shape: (2, num_test_non_edges)
            Test non-edges.
    """
    # Convert data to a scipy.sparse.csr_matrix representing the adjacency matrix of the graph
    A = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                     sparse_sizes=(data.num_nodes, data.num_nodes)).to_scipy(layout='csr')

    # Perform the split of the adjacency matrix into training, validation and test edges,
    # and validation and test non-edges
    train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05,
                                                                                            seed=seed, neg_mul=1,
                                                                                            every_node=True,
                                                                                            connected=True,
                                                                                            undirected=True,
                                                                                            use_edge_cover=True,
                                                                                            set_ops=True,
                                                                                            asserts=True)

    # Convert train_ones, val_ones, val_zeros, test_ones and test_zeros to torch.LongTensors
    train_edge_index = torch.from_numpy(train_ones.T.astype(int))
    val_edge_index = torch.from_numpy(val_ones.T.astype(int))
    val_non_edge_index = torch.from_numpy(val_zeros.T.astype(int))
    test_edge_index = torch.from_numpy(test_ones.T.astype(int))
    test_non_edge_index = torch.from_numpy(test_zeros.T.astype(int))

    return train_edge_index, val_edge_index, val_non_edge_index, test_edge_index, test_non_edge_index
