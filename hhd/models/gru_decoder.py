"""
GRU-based decoder for heavy-hexagon syndrome sequences.

Architecture
------------
Branched_GRU
    A unidirectional GRU encoder that takes a variable-length syndrome
    sequence together with the initial and final ancilla measurement
    vectors, then passes the last hidden state through a stack of
    residual MLP blocks (ResBlock1D) to produce a single binary logit.

    Input tensors
    ~~~~~~~~~~~~~
    seq            : (B, T, input_size)   -- syndrome sequence (packed internally)
    final_syndrome : (B, z_stab)          -- final round ancilla measurements
    initial_det    : (B, z_stab)          -- first round ancilla measurements
    lengths        : (B,)                 -- actual sequence lengths before padding

    Output
    ~~~~~~
    logits : (B, 1)  -- raw (un-sigmoided) binary logit; use BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class ResBlock1D(nn.Module):
    """Residual MLP block operating on a single feature vector per sample.

    Input / Output shape: (B, dim)

    Parameters
    ----------
    dim : int
        Feature dimension (must match the GRU output size).
    hidden_dim : int, optional
        Internal projection width.  Defaults to ``max(64, dim // 2)``.
    dropout : float
        Dropout probability applied after the residual add (default 0.0).
    """

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(64, dim // 2)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.fc1(x)
        y = F.gelu(self.norm1(y))
        y = self.fc2(y)
        y = self.norm2(y)
        y = self.drop(y)
        return F.gelu(y + identity)


class Branched_GRU(nn.Module):
    """GRU encoder with residual MLP head for binary logical-error classification.

    Parameters
    ----------
    input_size : int
        Number of detectors per syndrome round (``dpr``).
    hidden_size : int
        GRU hidden-state dimensionality.
    z_stab : int
        Number of Z-type stabilizers (``num_z``); determines the width of
        the initial/final ancilla vectors concatenated to each time step.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability used inside multi-layer GRU and ResBlock1D (default 0.3).
    bidirectional : bool
        Kept for API compatibility; always forced to ``False`` (unidirectional).
    res_blocks : int
        Number of ResBlock1D layers in the MLP head (default 2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        z_stab: int,
        num_layers: int,
        dropout: float = 0.3,
        bidirectional: bool = False,
        res_blocks: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False  # unidirectional only
        self.z_stab = z_stab
        self.dropout_p = dropout

        # Each time step receives [initial_det | syndrome | final_det]
        self.gru = nn.GRU(
            input_size=input_size + 2 * z_stab,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Residual MLP head
        blocks = []
        for _ in range(max(1, res_blocks)):
            blocks.append(ResBlock1D(hidden_size, hidden_dim=hidden_size, dropout=dropout))
        self.res_head = nn.Sequential(*blocks)

        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        seq: torch.Tensor,
        final_syndrome: torch.Tensor,
        initial_det: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        seq : torch.Tensor
            Shape ``(B, T, input_size)`` — padded syndrome sequence.
        final_syndrome : torch.Tensor
            Shape ``(B, z_stab)`` — final ancilla measurements.
        initial_det : torch.Tensor
            Shape ``(B, z_stab)`` — initial ancilla measurements.
        lengths : torch.Tensor
            Shape ``(B,)`` — true sequence lengths (before padding).

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)`` — raw logits.
        """
        B, T, _ = seq.size()

        # Broadcast initial/final syndrome to every time step
        initial_rep = initial_det.unsqueeze(1).expand(B, T, -1)
        final_rep = final_syndrome.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([initial_rep, seq, final_rep], dim=2)

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_last = h_n[-1]  # (B, hidden_size) — last layer's final hidden state

        feat = self.res_head(h_last)
        g = self.drop(feat)
        logits = self.classifier(g)  # (B, 1)
        return logits
