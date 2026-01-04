# models/transformer.py
from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Manual implementation of multi-head self-attention using nn.Linear primitives.
    Input: (B, S, E)
    Output: (B, S, E)
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # projections for q,k,v and final out
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reshape_for_heads(self, x: torch.Tensor):
        # x: (B, S, E) -> (B, num_heads, S, head_dim)
        B, S, E = x.size()
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, E)
        B, S, E = x.size()
        q = self.q_proj(x)  # (B, S, E)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qh = self.reshape_for_heads(q)  # (B, H, S, D)
        kh = self.reshape_for_heads(k)
        vh = self.reshape_for_heads(v)

        # scaled dot-product
        # attn_scores: (B, H, S, S)
        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # mask handling:
        # mask can be:
        #  - causal: (S, S) with -inf above diagonal (float), or
        #  - boolean padding mask of shape (B, S) that marks paddings True, or
        #  - combined: (B, S, S)
        if mask is not None:
            if mask.dtype == torch.bool:
                # mask True means masked out. Expand to (B,1,1,S) or (B,1,S,S)
                if mask.dim() == 2:
                    # (B,S) padding mask -> we want to mask key positions
                    key_mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
                    attn_scores = attn_scores.masked_fill(key_mask, float('-inf'))
                elif mask.dim() == 3:
                    # (B,S,S)
                    attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            else:
                # assume mask is additive float mask (S, S) or (B, S, S); add directly
                attn_scores = attn_scores + mask.unsqueeze(0) if mask.dim() == 2 else attn_scores + mask.unsqueeze(1)

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, S, S)
        attn_out = torch.matmul(attn_weights, vh)  # (B, H, S, D)
        # combine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, E)
        out = self.out_proj(attn_out)
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encodings (non-learnable) -> better extrapolation.
    """
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            # last odd dimension
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # register buffer so it moves with .to(device)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        S = x.size(1)
        x = x + self.pe[:S, :].unsqueeze(0).to(x.device)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Pre-LN Transformer Encoder Layer (Norm -> Attention -> Dropout -> Residual).
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN
        y = self.ln1(x)
        y = self.attn(y, mask=mask)
        y = self.dropout1(y)
        x = x + y

        z = self.ln2(x)
        z = self.mlp(z)
        z = self.dropout2(z)
        x = x + z
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of TransformerEncoderLayer with embedding + positional encoding.
    Expects vocab_size that does NOT include pad token -> we construct vocab_size+1 embedding.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 mlp_dim: int,
                 num_heads: int,
                 num_layers: int,
                 pad_token: int,
                 max_len: int,
                 dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=pad_token)
        self.pos = PositionalEncoding(embed_dim, max_len)
        self.input_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                                     for _ in range(num_layers)])
        self.pad_token = pad_token
        self.embed_dim = embed_dim

    def make_padding_mask(self, x: torch.Tensor):
        # x: (B, S)
        return (x == self.pad_token)  # boolean mask where True => padding

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S)
        B, S = x.size()
        x = self.embed(x)  # (B, S, E)
        x = self.pos(x)
        x = self.input_dropout(x)

        # build padding mask (B, S) -> we will pass boolean masks to layers which handle them
        pad_mask = self.make_padding_mask(x=torch.zeros_like(x[..., 0], dtype=torch.long))  # placeholder to keep type
        # but better to compute from input tokens; try to recover tokens if available
        # For safety: expecting embedding padding_idx is set, if tokens unknown, user should pass mask externally.
        # We'll accept `mask` param as additive mask or boolean masks and forward it to MHA.

        attn_mask = mask  # pass-through; higher-level module constructs appropriate causal mask + padding if needed

        out = x
        for layer in self.layers:
            out = layer(out, mask=attn_mask)
        return out  # (B, S, E)


class TransformerClassifier(nn.Module):
    """
    Encoder stack + final LayerNorm + linear projection.
    For sequence-level representation we use masked mean pooling by default.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 mlp_dim: int,
                 num_heads: int,
                 num_layers: int,
                 num_classes: int,
                 pad_token: int,
                 max_len: int,
                 dropout: float):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                          embed_dim=embed_dim,
                                          mlp_dim=mlp_dim,
                                          num_heads=num_heads,
                                          num_layers=num_layers,
                                          pad_token=pad_token,
                                          max_len=max_len,
                                          dropout=dropout)
        self.final_ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.proj.weight)
        self.pad_token = pad_token

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> torch.Tensor:
        """
        x: (B, S)
        lengths: (B,)
        mask: optional mask (causal additive mask or boolean masks)
        returns: logits (B, num_classes)
        """
        enc = self.encoder(x, lengths, mask=mask)  # (B, S, E)
        enc = self.final_ln(enc)

        # masked mean pooling
        B, S, E = enc.size()
        device = enc.device
        # Build padding mask from lengths
        seq_range = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
        mask_pad = seq_range >= lengths.unsqueeze(1)  # True at padded positions
        valid = (~mask_pad).unsqueeze(-1)  # (B, S, 1)
        enc_masked = enc * valid.to(enc.dtype)
        denom = valid.sum(dim=1).clamp(min=1).to(enc.dtype)  # (B, 1)
        pooled = enc_masked.sum(dim=1) / denom  # (B, E)

        logits = self.proj(pooled)  # (B, num_classes)
        return logits
