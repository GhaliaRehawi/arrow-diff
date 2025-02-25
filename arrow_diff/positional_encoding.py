import torch

from typing import TypeVar


# Create TypeVars for FloatTensor and LongTensor
FloatTensor = TypeVar('FloatTensor', torch.FloatTensor, torch.cuda.FloatTensor)
LongTensor = TypeVar('LongTensor', torch.LongTensor, torch.cuda.LongTensor)


def positional_encoding(position: LongTensor, dim: int) -> FloatTensor:
    """
    Computes positional encodings based on the formula in Vaswani et al. [1].

    [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin.
        "Attention Is All You Need". NeurIPS, 2017.

    Args:
        position: LongTensor, shape: (num_positions,)
            Positions for which to compute the positional encodings.
        dim: int
            Dimensionality of the positional encodings.

    Returns:
        encoding: FloatTensor, shape: (num_positions, d_model)
            Positional encodings of size dim for every position.
    """
    encoding = []

    for i in range(dim // 2):
        values = position / 10000 ** (2 * i / dim)

        encoding.append(torch.sin(values))
        encoding.append(torch.cos(values))

    encoding = torch.stack(encoding, dim=1)

    return encoding
