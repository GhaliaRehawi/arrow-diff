import torch
import torch.nn.functional as F

from abc import abstractmethod
from torch.nn import Module, ModuleList, Sequential, Embedding, Conv1d, Linear, GroupNorm, ReLU, MaxPool1d, LogSoftmax
from typing import List, Tuple, TypeVar


# Create TypeVars for FloatTensor and LongTensor
FloatTensor = TypeVar('FloatTensor', torch.FloatTensor, torch.cuda.FloatTensor)
LongTensor = TypeVar('LongTensor', torch.LongTensor, torch.cuda.LongTensor)


class TimeStepBlock(Module):
    """
    Time step block:
    Any module where forward() takes time step embeddings as a second argument.

    Code taken from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py
    """
    @abstractmethod
    def forward(self, x: FloatTensor, time_embeddings: FloatTensor) -> FloatTensor:
        """
        Forward pass of TimeStepBlock. Apply the module to x given time_embeddings time step embeddings.

        Args:
            x: FloatTensor
                Input tensor.
            time_embeddings: FloatTensor
                Time step embeddings.

        Returns:
            x: FloatTensor
                Output tensor.
        """
        pass


class TimeStepEmbeddingSequential(Sequential, TimeStepBlock):
    """
    Time step embedding Sequential:
    A sequential module that passes timestep embeddings to the children that support it as an extra input.

    Code taken from https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py
    """
    def forward(self, x: FloatTensor, time_embeddings: FloatTensor) -> FloatTensor:
        """
        Forward pass of TimeStepEmbeddingSequential.

        Args:
            x: FloatTensor
                Input tensor.
            time_embeddings: FloatTensor
                Time step embeddings.

        Returns:
            x: FloatTensor
                Output tensor.
        """
        for layer in self:
            if isinstance(layer, TimeStepBlock):
                x = layer(x, time_embeddings)
            else:
                x = layer(x)

        return x


class ResizeConv(Module):
    """
    Resize convolution.

    Code based on blog entry https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        """
        Initialization of ResizeConv.

        Args:
            in_channels: int
                Number of input channels.
            out_channels: int
                Number of output channels.
            kernel_size: int (optional, default: 3)
                Kernel size of the 1D convolution.
        """
        super().__init__()

        # Initialize a convolution layer to bring the input to out_channels channels
        self.conv = Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Forward pass of ResizeConv.

        Args:
            x: FloatTensor, shape: (batch_size, in_channels, num_features)
                Input tensor.

        Returns:
            x: FloatTensor, shape: (batch_size, out_channels, 2 * num_features)
                Output tensor.
        """
        # Upscale x by a factor of 2 using nearest-neighbor interpolation
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # Apply self.conv on x
        x = self.conv(x)

        return x


class ResidualBlock(TimeStepBlock):
    """
    Residual block.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, kernel_size: int = 3) -> None:
        """
        Initialization of ResidualBlock.

        Args:
            in_channels: int
                Number of input channels.
            out_channels: int
                Number of output channels.
            time_embedding_dim: int
                Dimensionality of the time step embeddings.
            kernel_size: int (optional, default: 3)
                Kernel size of the 1D convolution.
        """
        super().__init__()

        # Initialize a linear layer to scale the time embeddings to the correct dimensionality
        self.time_embedding_layers = Sequential(
            Linear(time_embedding_dim, out_channels),
            ReLU()
        )

        # Initialize a convolution block to scale the input to out_channel channels
        self.in_layers = Sequential(
            GroupNorm(in_channels // 4, in_channels),
            ReLU(),
            Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        )

        # Initialize a convolution block to transform the combined input and time embeddings
        self.out_layers = Sequential(
            GroupNorm(out_channels // 4, out_channels),
            ReLU(),
            Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        )

        # Initialize a convolution layer to scale the tensor in the skip-connection to out_channels channels
        self.skip_conv = Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x: FloatTensor, time_embeddings: FloatTensor) -> FloatTensor:
        """
        Forward pass of ResidualBlock.

        Args:
            x: FloatTensor, shape: (batch_size, in_channels, num_features)
                Input tensor.
            time_embeddings: FloatTensor, shape: (batch_size, time_embedding_dim)
                Time step embeddings.

        Returns:
            z: FloatTensor, shape: (batch_size, out_channels, num_features)
                Output tensor.
        """
        # Scale time_embeddings to the correct numer of channels
        time_embeddings = self.time_embedding_layers(time_embeddings)

        # Apply self.in_layers
        z = self.in_layers(x)

        # Add time_embeddings to z
        z = z + time_embeddings.unsqueeze(dim=2)

        # Apply self.out_layers
        z = self.out_layers(z)

        # Add the skip-connection
        z = z + self.skip_conv(x)

        return z


class ContractingBlock(Module):
    """
    Contracting block for the U-Net.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, num_res_blocks: int = 1,
                 kernel_size: int = 3) -> None:
        """
        Initialization of ContractingBlock.

        Args:
            in_channels: int
                Number of input channels.
            out_channels: int
                Number of output channels.
            time_embedding_dim: int
                Dimensionality of the time step embeddings.
            num_res_blocks: int (optional, default: 1)
                Number of residual blocks in a row inside ContractingBlock.
            kernel_size: int (optional, default: 3)
                Kernel size of the 1D convolution.
        """
        super().__init__()

        # Initialize the residual blocks
        res_blocks = []

        for i in range(num_res_blocks):
            res_blocks.append(ResidualBlock(in_channels, out_channels, time_embedding_dim, kernel_size=kernel_size))
            in_channels = out_channels

        self.res_blocks = TimeStepEmbeddingSequential(*res_blocks)

        # Initialize the max-pool layer
        self.max_pool = MaxPool1d(2)

    def forward(self, x: FloatTensor, time_embeddings: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Forward pass of ContractingBlock.

        Args:
            x: FloatTensor, shape: (batch_size, in_channels, num_features)
                Input tensor.
            time_embeddings: FloatTensor, shape: (batch_size, time_embedding_dim)
                Time step embeddings.

        Returns:
            x_downscaled: FloatTensor, shape: (batch_size, out_channels, num_features // 2)
                Downscaled output tensor.
            x: FloatTensor, shape: (batch_size, out_channels, num_features)
                Output tensor.
        """
        # Pass x and time_embeddings through the residual blocks
        x = self.res_blocks(x, time_embeddings)

        return self.max_pool(x), x


class ExpansiveBlock(Module):
    """
    Expansive block for the U-Net.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, num_res_blocks: int = 1,
                 kernel_size: int = 3) -> None:
        """
        Initialization of ExpansiveBlock.

        Args:
            in_channels: int
                Number of input channels.
            out_channels: int
                Number of output channels.
            time_embedding_dim: int
                Dimensionality of the time step embeddings.
            num_res_blocks: int (optional, default: 1)
                Number of residual blocks in a row inside ExpansiveBlock.
            kernel_size: int (optional, default: 3)
                Kernel size of the 1D convolution.
        """
        super().__init__()

        # Initialize the resize convolution
        self.resize_conv = ResizeConv(in_channels, out_channels, kernel_size=kernel_size)

        # Initialize the residual blocks
        res_blocks = []

        for i in range(num_res_blocks):
            res_blocks.append(ResidualBlock(in_channels, out_channels, time_embedding_dim, kernel_size=kernel_size))
            in_channels = out_channels

        self.res_blocks = TimeStepEmbeddingSequential(*res_blocks)

    def forward(self, copy: FloatTensor, x: FloatTensor, time_embeddings: FloatTensor) -> FloatTensor:
        """
        Forward pass of ExpansiveBlock.

        Args:
            copy: FloatTensor, shape: (batch_size, out_channels, 2 * num_features)
                Copied tensor from the contracting path from skip-connection.
            x: FloatTensor, shape: (batch_size, in_channels, num_features)
                Input tensor.
            time_embeddings: FloatTensor, shape: (batch_size, time_embedding_dim)
                Time step embeddings.
        Returns:
            x: FloatTensor, shape: (batch_size, out_channels, 2 * num_features)
                Output tensor.
        """
        # Upscale x with a resize convolution
        x = self.resize_conv(x)

        # Concatenate copy and x along the channel dimension
        x = torch.cat([copy, x], dim=1)

        # Pass x and time_embeddings through the residual blocks
        x = self.res_blocks(x, time_embeddings)

        return x


class UNet(Module):
    """
    U-Net architecture.
    """
    def __init__(self, in_channels: int, hidden_channels: List[int], num_time_steps: int, time_embedding_dim: int,
                 num_res_blocks: int = 1, kernel_size: int = 3) -> None:
        """
        Initialization of UNet.

        Args:
            in_channels: int
                Number of input channels (number of nodes).
            hidden_channels: List[int]
                Hidden channels for every level of the U-Net.
            num_time_steps: int
                Total number of time steps.
            time_embedding_dim: int
                Dimensionality of the time step embeddings.
            num_res_blocks: int (optional, default: 1)
                Number of residual blocks in a row inside ContractingBlock and ExpansiveBlock.
            kernel_size: int (optional, default: 3)
                Kernel size of the 1D convolution.
        """
        super().__init__()

        # Initialize learnable time embeddings
        self.time_embeddings = Embedding(num_time_steps, time_embedding_dim)

        out_channels = in_channels

        # Initialize ContractingBlocks for the contracting path
        self.contracting_blocks = ModuleList()

        for i in range(len(hidden_channels)):
            self.contracting_blocks.append(ContractingBlock(in_channels, hidden_channels[i], time_embedding_dim,
                                                            num_res_blocks=num_res_blocks, kernel_size=kernel_size))

            in_channels = hidden_channels[i]

        # Initialize the bottleneck block
        self.bottleneck_block = ResidualBlock(in_channels, 2 * in_channels, time_embedding_dim, kernel_size=kernel_size)

        # Initialize ExpansiveBlocks for the expansive path
        self.expansive_blocks = ModuleList()

        for i in range(len(hidden_channels) - 1, -1, -1):
            self.expansive_blocks.append(ExpansiveBlock(2 * in_channels, hidden_channels[i], time_embedding_dim,
                                                        num_res_blocks=num_res_blocks, kernel_size=kernel_size))

            in_channels = hidden_channels[i - 1]

        # Output convolution
        self.out_conv = Sequential(
            GroupNorm(hidden_channels[0] // 4, hidden_channels[0]),
            ReLU(),
            Conv1d(hidden_channels[0], out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        )

    def forward(self, x: FloatTensor, time_steps: LongTensor) -> FloatTensor:
        """
        Forward pass of UNet.

        Args:
            x: FloatTensor, shape: (batch_size, in_channels, num_features)
                Input tensor.
            time_steps: LongTensor, shape: (batch_size,)
                Time steps.

        Returns:
            x: FloatTensor, shape: (batch_size, in_channels, num_features)
        """
        # Get time embeddings for time_steps
        time_embeddings = self.time_embeddings(time_steps)

        copies = []

        # Contracting path
        for i in range(len(self.contracting_blocks)):
            x, copy = self.contracting_blocks[i](x, time_embeddings)

            # Save copy for skip-connection
            copies.append(copy)

        # Bottleneck
        x = self.bottleneck_block(x, time_embeddings)

        # Expansive path
        for i in range(len(self.expansive_blocks)):
            x = self.expansive_blocks[i](copies.pop(), x, time_embeddings)

        # Output convolution
        x = self.out_conv(x)

        return x



class UNetAdapter(Module):
    """
    Network that works as an adapter for the U-Net by projecting down the one-hot inputs to lower dimensional embedding
    space before applying the U-Net.
    """
    def __init__(self, hidden_channels: List[int], num_nodes: int, node_embedding_dim: int,
                 num_time_steps: int, time_embedding_dim: int, num_res_blocks: int = 1, kernel_size: int = 3) -> None:
        """
        Initialization of UNetAdapter.

        Args:
            hidden_channels: List[int]
                Hidden channels for every level of the U-Net.
            num_nodes: int (optional, default: None)
                Number of nodes in the graph.
            node_embedding_dim: int
                Dimensionality of the node embeddings.
            num_time_steps: int
                Total number of time steps.
            time_embedding_dim: int
                Dimensionality of the time step embeddings.
            num_res_blocks: int (optional, default: 1)
                Number of residual blocks in a row inside ContractingBlock and ExpansiveBlock.
            kernel_size: int (optional, default: 3)
                Kernel size of the 1D convolution.
        """
        super().__init__()

        # Initialize U-Net
        self.unet = UNet(node_embedding_dim, hidden_channels, num_time_steps, time_embedding_dim,
                         num_res_blocks=num_res_blocks, kernel_size=kernel_size)

        # Initialize node embeddings for all nodes and the mask state
        self.node_embeddings = Embedding(num_nodes + 1, node_embedding_dim)

        # Initialize ReLU and a linear layer
        self.linear = Sequential(
            ReLU(),
            Linear(node_embedding_dim, num_nodes)
        )

        # Initialize the log-softmax layer
        self.log_softmax = LogSoftmax(dim=2)

    def forward(self, x: LongTensor, time_steps: LongTensor) -> FloatTensor:
        """
        Forward pass of UNet.

        Args:
            x: LongTensor, shape: (batch_size, walk_length)
                Input tensor, where walk_length corresponds to in_channels.
            time_steps: LongTensor, shape: (batch_size,)
                Time steps.

        Returns:
            x: FloatTensor, shape: (batch_size, walk_length, num_nodes)
        """
        # Obtain node embeddings for the random walks
        x = self.node_embeddings(x)

        # Transpose dimensions 1 and 2
        x = torch.transpose(x, 1, 2)

        # Put x with time_steps through the U-Net
        x = self.unet(x, time_steps)

        # Transpose dimensions 1 and 2
        x = torch.transpose(x, 1, 2)

        # Use a linear layer to project the logits up to the number of nodes again
        x = self.linear(x)

        # Compute softmax
        x = self.log_softmax(x)

        return x
