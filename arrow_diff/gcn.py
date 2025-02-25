import torch
import torch.nn.functional as F

from torch.nn import Module
from torch_geometric.nn.conv import GCNConv
from typing import TypeVar


# Create TypeVars for FloatTensor and LongTensor
FloatTensor = TypeVar('FloatTensor', torch.FloatTensor, torch.cuda.FloatTensor)
LongTensor = TypeVar('LongTensor', torch.LongTensor, torch.cuda.LongTensor)


class GCN(Module):
    """
    Graph Convolutional Network (GCN) [1].

    [1] Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph Convolutional Networks". ICLR, 2017.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        """
        Initialization of GCN.

        Args:
            in_channels: int
                Dimensionality of the input node features.
            hidden_channels: int
                Dimensionality of the hidden features.
            out_channels: int
                Dimensionality of the output node features.
        """
        super().__init__()
        self.conv_1 = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: FloatTensor, edge_index: LongTensor) -> FloatTensor:
        """
        Forward pass of GCN.

        Args:
            x: torch.FloatTensor, shape: (num_nodes, in_channels)
                Input node features.
            edge_index: torch.LongTensor, shape: (2, num_edges)
                Indices representing the edges of the graph.

        Returns:
            x: torch.FloatTensor, shape: (num_nodes, out_channels)
                Node embeddings.
        """
        x = F.relu(self.conv_1(x, edge_index))
        x = self.conv_2(x, edge_index)
        return x
