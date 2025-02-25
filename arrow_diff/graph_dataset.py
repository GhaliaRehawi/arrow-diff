import torch

from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import contains_self_loops, is_undirected, remove_self_loops, to_undirected
from typing import Optional, Tuple

from arrow_diff.dropout import dropout_edge


class GraphPerturbationDataset(Dataset):
    """
    Dataset that generates perturbed versions of an original graph.
    """
    def __init__(self, x: FloatTensor, edge_index: LongTensor, num_graphs: int, edge_dropout: float,
                 fake_edge_ratios: Tuple[float, float], seed: Optional[int] = None) -> None:
        self.x = x
        self.edge_index = edge_index
        self.num_graphs = num_graphs
        self.edge_dropout = edge_dropout
        self.fake_edge_ratios = fake_edge_ratios
        self.seed = seed

        self.generator = None if seed is None else torch.Generator()

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> Data:
        edge_index = self.edge_index

        if self.generator is not None:
            self.generator.manual_seed((idx + 1) * self.seed)

        # Remove original edges
        edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout, force_undirected=True, generator=self.generator)

        num_valid_edges = edge_index.size(1)

        if self.generator is not None:
            self.generator.manual_seed((idx + 1) * self.seed)

        fake_edge_ratio = ((self.fake_edge_ratios[1] - self.fake_edge_ratios[0])
                           * torch.rand((), generator=self.generator).item() + self.fake_edge_ratios[0])

        # num_valid_edges // 2 because the edges are undirected and counted twice
        num_fake_edges = int(fake_edge_ratio * (num_valid_edges // 2))

        num_nodes = self.x.size(0)

        if self.generator is not None:
            self.generator.manual_seed((idx + 1) * self.seed)

        # Generate fake edges: Generate 2 * num_fake_edges, then filter duplicates and original edges
        edge_index_fake = torch.randint(num_nodes, (2, 2 * num_fake_edges), generator=self.generator)

        idxs = num_nodes * edge_index[0] + edge_index[1]
        idxs_fake = num_nodes * edge_index_fake[0] + edge_index_fake[1]

        idxs, counts = torch.unique(torch.cat([idxs, idxs_fake]), return_counts=True)

        # Get the indices of the unique edges in cat([edge_index, edge_index_fake])
        idxs = idxs[counts == 1]

        # Get the intersection of idxs with idxs_fake
        idxs, counts = torch.unique(torch.cat([idxs, idxs_fake]), return_counts=True)
        idxs_fake = idxs[counts == 2]

        row = torch.floor_divide(idxs_fake, num_nodes)
        col = torch.remainder(idxs_fake, num_nodes)

        edge_index_fake = torch.stack([row, col], dim=0)

        if contains_self_loops(edge_index_fake):
            edge_index_fake, _ = remove_self_loops(edge_index_fake)

        if edge_index_fake.size(1) < num_fake_edges:
            raise Warning(f'Only {edge_index_fake.size(1)}/{num_fake_edges} fake edges are added to the graph.')
        else:
            edge_index_fake = edge_index_fake[:, :num_fake_edges]

        if not is_undirected(edge_index_fake, num_nodes=num_nodes):
            edge_index_fake = to_undirected(edge_index_fake, num_nodes=num_nodes)

        num_fake_edges = edge_index_fake.size(1)

        # Add fake edges
        edge_index = torch.cat([edge_index, edge_index_fake], dim=1)

        # Create a label for each edge, whether it is valid (= 1) or fake (= 0)
        y = 1 - torch.repeat_interleave(torch.tensor([num_valid_edges, num_fake_edges]))

        data = Data(x=self.x, edge_index=edge_index, y=y.float())

        return data
