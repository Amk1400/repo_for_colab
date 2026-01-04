# models/recurrent.py
from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentCell(nn.Module):
    """
    A simple GRU-like cell implemented from linear layers.
    Not using nn.GRUCell (implemented from primitives).
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Update gate z, Reset gate r, Candidate h~
        self.lin_xz = nn.Linear(input_dim, hidden_dim, bias=True)
        self.lin_hz = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.lin_xr = nn.Linear(input_dim, hidden_dim, bias=True)
        self.lin_hr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.lin_xh = nn.Linear(input_dim, hidden_dim, bias=True)
        self.lin_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim), h_prev: (B, hidden_dim)
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h_prev))
        r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h_prev))
        h_tilde = torch.tanh(self.lin_xh(x) + self.lin_hh(r * h_prev))
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new


class RecurrentEncoder(nn.Module):
    """
    Stacked recurrent encoder using RecurrentCell.
    Supports bidirectional operation by running two separate stacks (forward/backward).
    Implements per-time-step "effective batch" skipping: at each time step we only run
    updates for samples that are still active (length > t).
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 pad_token: int,
                 bidirectional: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.pad_token = pad_token
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_token)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # forward layers
        self.fwd_cells = nn.ModuleList([RecurrentCell(embed_dim if i == 0 else hidden_dim, hidden_dim)
                                        for i in range(num_layers)])
        # backward layers if needed
        if self.bidirectional:
            self.bwd_cells = nn.ModuleList([RecurrentCell(embed_dim if i == 0 else hidden_dim, hidden_dim)
                                            for i in range(num_layers)])
        else:
            self.bwd_cells = None

    def forward_single_direction(self, embeds: torch.Tensor, lengths: torch.Tensor, cells: nn.ModuleList, reverse: bool = False):
        """
        embeds: (B, S, D)
        lengths: (B,)
        returns: outputs (B, S, H)
        """
        B, S, D = embeds.size()
        H = self.hidden_dim
        device = embeds.device

        # initialize hidden states per layer
        hs = [torch.zeros(B, H, device=device) for _ in range(self.num_layers)]
        outputs = embeds.new_zeros(B, S, H)

        if reverse:
            time_range = range(S - 1, -1, -1)
        else:
            time_range = range(0, S)

        # convert lengths to CPU for comparisons (fast)
        lengths_cpu = lengths.detach().cpu()

        for t in time_range:
            # compute mask of samples active at time t
            if reverse:
                # active if (length - 1 - t) >= 0  -> length > (S - t - 1) equivalently use index mapping:
                # simpler: active if t < S and (S - t) <= lengths -> but we can compute idx per batch
                active_mask = (lengths_cpu > (S - 1 - t)).to(device)  # bool tensor
            else:
                active_mask = (lengths_cpu > t).to(device)

            if not active_mask.any():
                # nothing to do at this time step
                continue

            # Select active indices
            active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            x_t = embeds[active_idx, (t if not reverse else t), :]  # (A, D)

            # propagate through stacked cells
            for layer_idx, cell in enumerate(cells):
                h_prev = hs[layer_idx][active_idx]  # (A, H)
                h_new = cell(x_t, h_prev)  # (A, H)
                # write back
                hs[layer_idx][active_idx] = h_new
                # prepare input for next layer
                x_t = self.dropout(h_new)

            # store top-layer hidden to outputs
            outputs[active_idx, (t if not reverse else t), :] = hs[-1][active_idx]

        return outputs  # (B, S, H)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S)
        lengths: (B,)
        returns:
            if bidirectional -> (B, S, 2*H)
            else -> (B, S, H)
        """
        embeds = self.embed(x)  # (B, S, D)
        fwd_out = self.forward_single_direction(embeds, lengths, self.fwd_cells, reverse=False)

        if self.bidirectional:
            bwd_out = self.forward_single_direction(embeds, lengths, self.bwd_cells, reverse=True)
            out = torch.cat([fwd_out, bwd_out], dim=-1)
        else:
            out = fwd_out
        return out  # (B, S, H or 2H)


class RecurrentClassifier(nn.Module):
    """
    Builds on RecurrentEncoder. For sequence-to-sequence outputs (modular addition)
    we provide logits per time-step; for classification tasks (palindrome) we support
    return_last_step_only=True -> (B, num_classes).
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 pad_token: int,
                 bidirectional: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = RecurrentEncoder(vocab_size=vocab_size,
                                        embed_dim=embed_dim,
                                        hidden_dim=hidden_dim,
                                        num_layers=num_layers,
                                        pad_token=pad_token,
                                        bidirectional=bidirectional,
                                        dropout=dropout)
        self.bidirectional = bidirectional
        out_dim = hidden_dim * (2 if bidirectional else 1)
        # final projection to classes (applied per-step or on pooled vector)
        self.classifier = nn.Linear(out_dim, num_classes)

        # initialization
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor,
                return_last_step_only: bool = True) -> torch.Tensor:
        """
        x: (B, S)
        lengths: (B,)
        return_last_step_only:
            - True -> returns (B, num_classes)
            - False -> returns (B, S, num_classes)
        """
        enc_out = self.encoder(x, lengths)  # (B, S, D)
        if return_last_step_only:
            B, S, D = enc_out.size()
            device = enc_out.device
            batch_idx = torch.arange(0, B, device=device)
            last_idx = (lengths - 1).to(device)
            last_hidden = enc_out[batch_idx, last_idx, :]  # (B, D)
            logits = self.classifier(last_hidden)  # (B, num_classes)
            return logits
        else:
            # per-step logits (for deep supervision)
            logits = self.classifier(enc_out)  # (B, S, num_classes)
            return logits
