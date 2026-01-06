# models/transformer_with_pool_and_training.py
from typing import *
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------
# Core model (with pooling)
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reshape_for_heads(self, x: torch.Tensor):
        B, S, E = x.size()
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attn: bool = False, mirror_bias: Optional[dict] = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self._core_attention(q, k, v, mask=mask, return_attn=return_attn, mirror_bias=mirror_bias)

    def forward_qkv(self, q_src: torch.Tensor, kv_src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                    return_attn: bool = False, mirror_bias: Optional[dict] = None):
        q = self.q_proj(q_src)
        k = self.k_proj(kv_src)
        v = self.v_proj(kv_src)
        return self._core_attention(q, k, v, mask=mask, return_attn=return_attn, mirror_bias=mirror_bias)

    def _core_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        mask: Optional[torch.Tensor] = None, return_attn: bool = False,
                        mirror_bias: Optional[dict] = None):
        B, S_q, E = q.size()
        S_k = k.size(1)
        qh = self.reshape_for_heads(q)
        kh = self.reshape_for_heads(k)
        vh = self.reshape_for_heads(v)

        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mirror_bias is not None:
            alpha = float(mirror_bias.get('alpha', 0.5))
            device = attn_scores.device
            i_idx = torch.arange(S_q, device=device).view(-1, 1)
            j_idx = torch.arange(S_k, device=device).view(1, -1)
            mirror_target = (S_k - 1) - i_idx
            dist = (j_idx - mirror_target).abs().float()
            bias = - alpha * (dist / float(max(1, S_k)))
            bias = bias.to(attn_scores.dtype)
            attn_scores = attn_scores + bias.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            if mask.dtype == torch.bool:
                if mask.dim() == 2:
                    key_mask = mask.unsqueeze(1).unsqueeze(2)
                    attn_scores = attn_scores.masked_fill(key_mask, float('-inf'))
                elif mask.dim() == 3:
                    attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
            else:
                if mask.dim() == 2:
                    attn_scores = attn_scores + mask.unsqueeze(0)
                else:
                    attn_scores = attn_scores + mask.unsqueeze(1)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, vh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, E)
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
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        x = x + self.pe[:S, :].unsqueeze(0).to(x.device)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    Encoder with gated symmetry, multi-scale pooling and mirror cross-attention.
    Supports forcing subsample factor for training on short sequences.
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
        self.max_len = max_len
        self.pos = PositionalEncoding(embed_dim, max_len)
        self.input_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                                     for _ in range(num_layers)])
        self.pad_token = pad_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # gated symmetry projection
        self.sym_mlp = nn.Linear(embed_dim * 4, embed_dim)
        nn.init.xavier_uniform_(self.sym_mlp.weight)
        self.sym_gate = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.sym_gate.weight)

        # projection after concat(forward,reverse)
        self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.xavier_uniform_(self.concat_proj.weight)

        # pool_proj
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.pool_proj.weight)

        # mirror attention
        self.mirror_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.last_subsample_factor = 1

    def _compute_subsample_factor(self, seq_len: int, target_post_len: int = 80) -> int:
        if seq_len <= target_post_len:
            return 1
        factor = (seq_len + target_post_len - 1) // target_post_len
        return int(max(1, factor))

    def _multi_scale_factors(self, factor: int) -> Tuple[int, int]:
        if factor <= 1:
            return 1, 1
        factor_small = max(1, factor // 2)
        return factor, factor_small

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None,
                force_subsample_factor: Optional[int] = None,
                force_mirror_alpha: Optional[float] = None) -> torch.Tensor:
        B, S = x.size()
        device = x.device

        x_emb = self.embed(x)  # (B, S, E)

        # gated symmetry
        x_rev_emb = x_emb.flip(dims=[1])
        diff = torch.abs(x_emb - x_rev_emb)
        prod = x_emb * x_rev_emb
        sym_in = torch.cat([x_emb, x_rev_emb, prod, diff], dim=-1)
        sym_mapped = self.sym_mlp(sym_in)
        gate = torch.sigmoid(self.sym_gate(sym_mapped))
        x_emb = x_emb + gate * sym_mapped

        # decide subsample factor
        computed_factor = self._compute_subsample_factor(S, target_post_len=80)
        factor = int(computed_factor if force_subsample_factor is None else max(1, int(force_subsample_factor)))
        self.last_subsample_factor = factor
        factor, factor_small = self._multi_scale_factors(factor)

        if factor > 1:
            # multi-scale pooling
            x_t = x_emb.transpose(1, 2)
            pad_len = (factor - (S % factor)) % factor
            if pad_len:
                pad_tensor = torch.zeros(B, self.embed_dim, pad_len, device=device, dtype=x_t.dtype)
                x_t = torch.cat([x_t, pad_tensor], dim=2)
            x_pooled = F.avg_pool1d(x_t, kernel_size=factor, stride=factor)
            x_fwd = x_pooled.transpose(1, 2)
            x_fwd = self.pool_proj(x_fwd)

            if factor_small != factor:
                pad_len_small = (factor_small - (S % factor_small)) % factor_small
                x_t_small = x_emb.transpose(1, 2)
                if pad_len_small:
                    pad_tensor_s = torch.zeros(B, self.embed_dim, pad_len_small, device=device, dtype=x_t_small.dtype)
                    x_t_small = torch.cat([x_t_small, pad_tensor_s], dim=2)
                x_pooled_small = F.avg_pool1d(x_t_small, kernel_size=factor_small, stride=factor_small)
                x_fwd_small = x_pooled_small.transpose(1, 2)
                x_fwd_small = self.pool_proj(x_fwd_small)
            else:
                x_fwd_small = None

            x_rev_emb2 = x_emb.flip(dims=[1])
            x_rt = x_rev_emb2.transpose(1, 2)
            if pad_len:
                pad_tensor_r = torch.zeros(B, self.embed_dim, pad_len, device=device, dtype=x_rt.dtype)
                x_rt = torch.cat([x_rt, pad_tensor_r], dim=2)
            x_r_pooled = F.avg_pool1d(x_rt, kernel_size=factor, stride=factor)
            x_rev = x_r_pooled.transpose(1, 2)
            x_rev = self.pool_proj(x_rev)

            x_cat = torch.cat([x_fwd, x_rev], dim=-1)
            x = self.concat_proj(x_cat)

            S_post = x.size(1)
            positions = torch.arange(0, S, step=factor, device=device)[:S_post].long()
            max_pos = min(self.max_len - 1, S - 1)
            positions = positions.clamp(0, max_pos)
            pe = self.pos.pe.to(device)
            pos_pe = pe[positions]
            x = x + pos_pe.unsqueeze(0)

            if x_fwd_small is not None:
                x_small = F.interpolate(x_fwd_small.transpose(1, 2), size=S_post, mode='linear', align_corners=False).transpose(1, 2)
                x = x + 0.5 * x_small

            new_lengths = ((lengths + factor - 1) // factor).to(lengths.device)
            lengths = new_lengths
        else:
            x = self.pos(x_emb)

        x = self.input_dropout(x)

        # mirror cross-attention
        if getattr(self, "mirror_attn", None) is not None:
            x_rev_post = x.flip(dims=[1])
            mirror_alpha = 0.5 if force_mirror_alpha is None else float(force_mirror_alpha)
            mirror_out = self.mirror_attn.forward_qkv(q_src=x, kv_src=x_rev_post, mask=None,
                                                      mirror_bias={'alpha': mirror_alpha})
            x = x + 0.7 * mirror_out

        out = x
        for layer in self.layers:
            out = layer(out, mask=mask)

        self.last_subsample_factor = factor
        return out


class TransformerClassifier(nn.Module):
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                force_subsample_factor: Optional[int] = None,
                force_mirror_alpha: Optional[float] = None) -> torch.Tensor:
        enc = self.encoder(x, lengths, mask=mask,
                           force_subsample_factor=force_subsample_factor,
                           force_mirror_alpha=force_mirror_alpha)
        enc = self.final_ln(enc)

        B, S_post, E = enc.size()
        device = enc.device

        factor = getattr(self.encoder, "last_subsample_factor", 1)
        if factor > 1:
            new_lengths = ((lengths + factor - 1) // factor).to(lengths.device)
        else:
            new_lengths = lengths.to(lengths.device)

        seq_range = torch.arange(0, S_post, device=device).unsqueeze(0).expand(B, S_post)
        mask_pad = seq_range >= new_lengths.unsqueeze(1)
        valid = (~mask_pad).unsqueeze(-1)
        enc_masked = enc * valid.to(enc.dtype)
        denom = valid.sum(dim=1).clamp(min=1).to(enc.dtype)
        pooled = enc_masked.sum(dim=1) / denom

        logits = self.proj(pooled)
        return logits

# ---------------------------
# Auxiliary loss (mirror matching)
# ---------------------------
def mirror_matching_loss_from_enc(enc: torch.Tensor,
                                  tokens: torch.LongTensor,
                                  lengths: torch.LongTensor,
                                  subsample_factor: int,
                                  num_pairs: int = 16,
                                  margin: float = 1.0) -> torch.Tensor:
    """
    enc: (B, S_post, E) encoder outputs AFTER pooling (or not).
    tokens: (B, S_original) original token indices.
    lengths: (B,) original lengths.
    subsample_factor: integer factor used to obtain enc length (1 means no pooling).
    Behavior: sample indices i from original token positions, map to pooled indices:
              pooled_i = floor(i / subsample_factor). Compare enc[b, pooled_i] with enc[b, pooled_j].
    """
    device = enc.device
    B, S_post, E = enc.size()
    loss = enc.new_zeros(1).sum()
    count = 0
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 0:
            continue
        # number of valid pairs per sequence
        for _ in range(num_pairs):
            i = np.random.randint(0, L)
            j = L - 1 - i
            pi = min(S_post - 1, i // subsample_factor)
            pj = min(S_post - 1, j // subsample_factor)
            hi = enc[b, pi]
            hj = enc[b, pj]
            same = (tokens[b, i].item() == tokens[b, j].item())
            if same:
                loss = loss + F.mse_loss(hi, hj)
            else:
                # push apart if too close
                dist = F.pairwise_distance(hi.unsqueeze(0), hj.unsqueeze(0), p=2)
                loss = loss + F.relu(margin - dist).mean()
            count += 1
    if count == 0:
        return loss
    return loss / count

# ---------------------------
# Training / evaluation helpers
# ---------------------------
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device,
                   batch_limit: Optional[int] = None,
                   verbose: bool = False) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    # also track per-length bands
    bands = {}  # band_range -> (correct, total)
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if batch_limit is not None and bi >= batch_limit:
                break
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            logits = model(x, lengths, force_subsample_factor=None)
            preds = logits.argmax(dim=-1)
            total += x.size(0)
            correct += (preds == y).sum().item()
            # banding by original lengths
            for i in range(x.size(0)):
                L = int(lengths[i].item())
                # define bands of width 50 (can change)
                band_start = (L // 50) * 50
                band = f"{band_start}-{band_start+49}"
                if band not in bands:
                    bands[band] = [0, 0]
                bands[band][1] += 1
                if preds[i].item() == y[i].item():
                    bands[band][0] += 1
    overall_acc = 0.0 if total == 0 else correct / total
    band_acc = {k: (v[0] / v[1] if v[1] > 0 else 0.0) for k, v in bands.items()}
    if verbose:
        print(f"Eval overall acc: {overall_acc:.4f}; bands: {band_acc}")
    return {"accuracy": overall_acc, "band_accuracy": band_acc}

def train_model(model: nn.Module,
                train_loader: DataLoader,
                dev_loader: Optional[DataLoader],
                test_loader: Optional[DataLoader],
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                num_epochs: int = 10,
                aux_weight: float = 0.1,
                pool_prob: float = 0.5,
                pool_choices: Sequence[int] = (2, 3, 4),
                aux_pairs: int = 16,
                log_every: int = 100):
    """
    Training loop that:
    - with probability pool_prob, forces a subsample factor sampled from pool_choices for the batch
      (so pooling modules are trained even with short sequences).
    - adds mirror matching auxiliary loss computed on the encoder outputs
    - evaluates on dev/test sets at end of each epoch
    """
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_cnt = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            x, y, lengths = batch
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)

            use_pool = (random.random() < pool_prob)
            factor = int(random.choice(pool_choices)) if use_pool else None

            # forward through encoder to get enc (we will compute logits manually to also compute aux)
            enc = model.encoder(x, lengths, mask=None, force_subsample_factor=factor)
            enc_ln = model.final_ln(enc)
            # compute pooled sequence-level representation as in model.forward
            B, S_post, E = enc_ln.size()
            factor_used = getattr(model.encoder, "last_subsample_factor", 1)
            if factor_used > 1:
                new_lengths = ((lengths + factor_used - 1) // factor_used).to(lengths.device)
            else:
                new_lengths = lengths.to(lengths.device)
            seq_range = torch.arange(0, S_post, device=device).unsqueeze(0).expand(B, S_post)
            mask_pad = seq_range >= new_lengths.unsqueeze(1)
            valid = (~mask_pad).unsqueeze(-1)
            enc_masked = enc_ln * valid.to(enc_ln.dtype)
            denom = valid.sum(dim=1).clamp(min=1).to(enc_ln.dtype)
            pooled = enc_masked.sum(dim=1) / denom
            logits = model.proj(pooled)

            loss_cls = criterion(logits, y)
            # auxiliary mirror matching loss computed on enc (post-pool)
            loss_aux = mirror_matching_loss_from_enc(enc, x.cpu(), lengths.cpu(), subsample_factor=factor_used,
                                                    num_pairs=aux_pairs)
            loss = loss_cls + aux_weight * loss_aux.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cnt += 1

            if batch_idx % log_every == 0:
                avg_loss = running_loss / max(1, running_cnt)
                print(f"[Epoch {epoch}] batch {batch_idx} avg_loss {avg_loss:.4f} (pool={'Y' if use_pool else 'N'}, factor={factor})")
                running_loss = 0.0
                running_cnt = 0

        t1 = time.time()
        print(f"Epoch {epoch} finished in {t1 - t0:.1f}s")

        # evaluate
        if dev_loader is not None:
            dev_stats = evaluate_model(model, dev_loader, device, verbose=True)
            print(f"Dev acc: {dev_stats['accuracy']:.4f}")
        if test_loader is not None:
            test_stats = evaluate_model(model, test_loader, device, verbose=True)
            print(f"Test acc: {test_stats['accuracy']:.4f}")

    return model

# ---------------------------
# Example usage (commented)
# ---------------------------
"""
# from models.transformer_with_pool_and_training import TransformerClassifier, train_model
# Assume you have train_ds, dev_ds, test_ds as PyTorch Dataset objects that return (seq_tensor, label, length)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

model = TransformerClassifier(vocab_size=VOCAB_SIZE, embed_dim=128, mlp_dim=256, num_heads=8,
                              num_layers=4, num_classes=2, pad_token=PAD_TOKEN, max_len=1024, dropout=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained = train_model(model, train_loader, dev_loader, test_loader, optimizer, criterion, device,
                      num_epochs=10, aux_weight=0.08, pool_prob=0.5, pool_choices=(2,3,4),
                      aux_pairs=8, log_every=200)
"""
