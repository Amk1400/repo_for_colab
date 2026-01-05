# models/transformer.py
from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Manual implementation of multi-head self-attention using nn.Linear primitives.
    Supports normal self-attention (forward) and cross-attention style usage via forward_qkv.
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
                mask: Optional[torch.Tensor] = None,
                return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard self-attention where q=k=v=project(x).
        If return_attn True, returns (out, attn_weights) where attn_weights shape is (B,H,S,S)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self._core_attention(q, k, v, mask=mask, return_attn=return_attn)

    def forward_qkv(self,
                    q_src: torch.Tensor,
                    kv_src: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cross-attention style: q is projected from q_src, k/v from kv_src.
        q_src: (B, S_q, E)
        kv_src: (B, S_kv, E)
        """
        q = self.q_proj(q_src)
        k = self.k_proj(kv_src)
        v = self.v_proj(kv_src)
        return self._core_attention(q, k, v, mask=mask, return_attn=return_attn)

    def _core_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        # q,k,v: (B, S, E) already projected
        B, S_q, E = q.size()
        S_k = k.size(1)
        qh = self.reshape_for_heads(q)  # (B, H, S_q, D)
        kh = self.reshape_for_heads(k)  # (B, H, S_k, D)
        vh = self.reshape_for_heads(v)  # (B, H, S_k, D)

        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,S_q,S_k)

        if mask is not None:
            if mask.dtype == torch.bool:
                # mask True -> masked out
                if mask.dim() == 2:
                    # (B, S_k) padding mask -> mask key positions
                    key_mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S_k)
                    attn_scores = attn_scores.masked_fill(key_mask, float('-inf'))
                elif mask.dim() == 3:
                    # (B, S_q, S_k)
                    attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            else:
                # additive float mask
                if mask.dim() == 2:
                    attn_scores = attn_scores + mask.unsqueeze(0)
                else:
                    attn_scores = attn_scores + mask.unsqueeze(1)

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B,H,S_q,S_k)
        attn_out = torch.matmul(attn_weights, vh)  # (B,H,S_q,D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, E)  # (B,S_q,E)
        out = self.out_proj(attn_out)
        if return_attn:
            return out, attn_weights
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
    Stores full pe buffer for max_len and allows indexing.
    """
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
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
    Auto-computes a subsample factor at runtime (no external config change).
    Enhancements:
      - gated symmetry channel (rich features from x and reversed x)
      - multi-scale pooling (two scales: factor and factor_small)
      - concat forward+reverse + concat_proj preserved
      - mirror (cross-) attention applied on pooled sequence to explicitly attend mirrored positions
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
        # store original max_len and create full positional encodings
        self.max_len = max_len
        self.pos = PositionalEncoding(embed_dim, max_len)
        self.input_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                                     for _ in range(num_layers)])
        self.pad_token = pad_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # gated symmetry projection: concat [x, rev, x*rev, |x-rev|] -> small MLP -> gated add
        self.sym_mlp = nn.Linear(embed_dim * 4, embed_dim)
        nn.init.xavier_uniform_(self.sym_mlp.weight)
        self.sym_gate = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.sym_gate.weight)

        # projection after concat(forward,reverse) -> embed_dim
        self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.xavier_uniform_(self.concat_proj.weight)

        # small learnable projection applied to pooled features (acts like pointwise conv after pooling)
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.pool_proj.weight)

        # mirror attention module (cross-attn q from forward, kv from reverse)
        self.mirror_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # last used subsample factor (defaults to 1)
        self.last_subsample_factor = 1

    def _compute_subsample_factor(self, seq_len: int, target_post_len: int = 80) -> int:
        """
        Compute an integer subsample factor to aim roughly for target_post_len.
        Ensures factor >=1.
        """
        if seq_len <= target_post_len:
            return 1
        # ceil division
        factor = (seq_len + target_post_len - 1) // target_post_len
        return int(max(1, factor))

    def _multi_scale_factors(self, factor: int) -> Tuple[int, int]:
        """
        return (factor, factor_small) where factor_small is a smaller pooling factor (>=1)
        """
        if factor <= 1:
            return 1, 1
        factor_small = max(1, factor // 2)
        return factor, factor_small

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S)
        B, S = x.size()
        device = x.device

        # Embedding
        x_emb = self.embed(x)  # (B, S, E)

        # --- Gated Symmetry channel: form rich features from forward and reversed embeddings ---
        x_rev_emb = x_emb.flip(dims=[1])  # (B, S, E)
        diff = torch.abs(x_emb - x_rev_emb)  # (B,S,E)
        prod = x_emb * x_rev_emb  # (B,S,E)
        # concat features
        sym_in = torch.cat([x_emb, x_rev_emb, prod, diff], dim=-1)  # (B,S,4E)
        sym_mapped = self.sym_mlp(sym_in)  # (B,S,E)
        gate = torch.sigmoid(self.sym_gate(sym_mapped))  # (B,S,E)
        # gated fusion
        x_emb = x_emb + gate * sym_mapped

        # decide subsample factor dynamically based on S and target
        factor = self._compute_subsample_factor(S, target_post_len=80)
        self.last_subsample_factor = factor
        factor, factor_small = self._multi_scale_factors(factor)

        if factor > 1:
            # ===== multi-scale pooling =====
            # forward pooled (large factor)
            x_t = x_emb.transpose(1, 2)  # (B, E, S)
            pad_len = (factor - (S % factor)) % factor
            if pad_len:
                pad_tensor = torch.zeros(B, self.embed_dim, pad_len, device=device, dtype=x_t.dtype)
                x_t = torch.cat([x_t, pad_tensor], dim=2)
            x_pooled = F.avg_pool1d(x_t, kernel_size=factor, stride=factor)  # (B, E, S_post)
            x_fwd = x_pooled.transpose(1, 2)  # (B, S_post, E)
            x_fwd = self.pool_proj(x_fwd)

            # forward pooled (small factor) -- only if different
            if factor_small != factor:
                pad_len_small = (factor_small - (S % factor_small)) % factor_small
                x_t_small = x_emb.transpose(1, 2)
                if pad_len_small:
                    pad_tensor_s = torch.zeros(B, self.embed_dim, pad_len_small, device=device, dtype=x_t_small.dtype)
                    x_t_small = torch.cat([x_t_small, pad_tensor_s], dim=2)
                x_pooled_small = F.avg_pool1d(x_t_small, kernel_size=factor_small, stride=factor_small)
                x_fwd_small = x_pooled_small.transpose(1, 2)  # (B, S_post_small, E)
                x_fwd_small = self.pool_proj(x_fwd_small)
            else:
                x_fwd_small = None

            # reversed pooled (mirrored windows) for large factor
            x_rev_emb2 = x_emb.flip(dims=[1])  # (B, S, E)
            x_rt = x_rev_emb2.transpose(1, 2)
            if pad_len:
                pad_tensor_r = torch.zeros(B, self.embed_dim, pad_len, device=device, dtype=x_rt.dtype)
                x_rt = torch.cat([x_rt, pad_tensor_r], dim=2)
            x_r_pooled = F.avg_pool1d(x_rt, kernel_size=factor, stride=factor)  # (B, E, S_post)
            x_rev = x_r_pooled.transpose(1, 2)  # (B, S_post, E)
            x_rev = self.pool_proj(x_rev)

            # concat forward and reverse pooled features (per position)
            x_cat = torch.cat([x_fwd, x_rev], dim=-1)  # (B, S_post, 2E)
            # project back to embed_dim
            x = self.concat_proj(x_cat)  # (B, S_post, E)

            # positions for pos encoding sampling
            S_post = x.size(1)
            positions = torch.arange(0, S, step=factor, device=device)[:S_post].long()  # (S_post,)
            max_pos = min(self.max_len - 1, S - 1)
            positions = positions.clamp(0, max_pos)
            pe = self.pos.pe.to(device)  # (max_len, E)
            pos_pe = pe[positions]  # (S_post, E)
            x = x + pos_pe.unsqueeze(0)

            # If multi-scale small exists, upsample/align small features and fuse (simple approach: project & sum)
            if x_fwd_small is not None:
                # we need to bring x_fwd_small to length S_post
                # simplest: average pool/upsample small->S_post by taking every (factor//factor_small)-th or interpolate
                # we'll use simple interpolation (linear) along sequence
                # x_fwd_small: (B, S_post_small, E)
                x_small = F.interpolate(x_fwd_small.transpose(1, 2), size=S_post, mode='linear', align_corners=False).transpose(1, 2)
                # fuse (sum) and re-project to stabilize (use concat_proj)
                x = x + 0.5 * x_small  # light fusion

            # update lengths (ceil div)
            new_lengths = ((lengths + factor - 1) // factor).to(lengths.device)
            lengths = new_lengths
        else:
            # no subsample: add standard first-S positional encodings on enriched embedding
            x = self.pos(x_emb)
            # lengths unchanged

        x = self.input_dropout(x)

        # ===== Mirror (cross-) attention: q from x, kv from reversed x =====
        # This explicitly provides direct i <-> mirrored(i) pathway
        # Build reversed sequence of x (post-pool)
        if getattr(self, "mirror_attn", None) is not None:
            x_rev_post = x.flip(dims=[1])  # (B, S_post, E)
            # cross-attend: q from x, kv from x_rev_post
            mirror_out = self.mirror_attn.forward_qkv(q_src=x, kv_src=x_rev_post, mask=None)
            # fuse mirror_out (residual-style). use a moderate scalar to not overwhelm self-attention path
            x = x + 0.7 * mirror_out

        out = x
        for layer in self.layers:
            out = layer(out, mask=mask)

        # store last factor
        self.last_subsample_factor = factor
        return out


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
        enc = self.encoder(x, lengths, mask=mask)  # (B, S_post, E)
        enc = self.final_ln(enc)

        B, S_post, E = enc.size()
        device = enc.device

        # compute post-subsample lengths using last_subsample_factor stored in encoder
        factor = getattr(self.encoder, "last_subsample_factor", 1)
        if factor > 1:
            new_lengths = ((lengths + factor - 1) // factor).to(lengths.device)
        else:
            new_lengths = lengths.to(lengths.device)

        seq_range = torch.arange(0, S_post, device=device).unsqueeze(0).expand(B, S_post)
        mask_pad = seq_range >= new_lengths.unsqueeze(1)  # True at padded positions
        valid = (~mask_pad).unsqueeze(-1)  # (B, S_post, 1)
        enc_masked = enc * valid.to(enc.dtype)
        denom = valid.sum(dim=1).clamp(min=1).to(enc.dtype)  # (B, 1)
        pooled = enc_masked.sum(dim=1) / denom  # (B, E)

        logits = self.proj(pooled)  # (B, num_classes)
        return logits
