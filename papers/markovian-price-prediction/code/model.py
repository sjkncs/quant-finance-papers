"""
model.py — MMRF model architecture: Markovian Multi-Resolution Forecasting.

Contains:
  - ResolutionTokenizer: per-resolution token embedding
  - CompressedSummaryNetwork: processes distant resolution summaries
  - CrossResolutionTransformer: shared transformer for multi-resolution tokens
  - MMRFModel: full model with sliding-window Markovian prediction
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ResolutionTokenizer(nn.Module):
    """Tokenizes features from a single resolution into token embeddings.

    Input: (B, seq_len, n_assets * n_features)
    Output: (B, seq_len, d_model)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        resolution_id: int = 0,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.res_emb = nn.Embedding(5, d_model)  # 5 resolution levels
        self.resolution_id = resolution_id
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        tokens = self.proj(x)  # (B, L, d_model)
        tokens = self.pos_enc(tokens)
        res_id = torch.full((B,), self.resolution_id, device=x.device, dtype=torch.long)
        tokens = tokens + self.res_emb(res_id).unsqueeze(1)
        return self.norm(tokens)


class CompressedSummaryNetwork(nn.Module):
    """Processes compressed statistical summaries from distant resolutions.

    Input: (B, n_distant_resolutions, summary_dim)
    Output: (B, n_summary_tokens, d_model)
    """

    def __init__(
        self,
        summary_dim: int = 30,
        d_model: int = 256,
        n_summary_tokens: int = 4,
    ):
        super().__init__()
        self.n_summary_tokens = n_summary_tokens
        self.proj = nn.Linear(summary_dim, d_model)
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_summary_tokens, d_model) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, summaries: torch.Tensor) -> torch.Tensor:
        """
        summaries: (B, n_res, summary_dim)
        Returns: (B, n_summary_tokens, d_model)
        """
        B = summaries.shape[0]
        summary_feats = self.proj(summaries)  # (B, n_res, d_model)

        # Cross-attend from learnable query tokens to summary features
        queries = self.query_tokens.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, summary_feats, summary_feats)
        tokens = self.norm(queries + attn_out)

        # Feedforward
        ff_out = self.ff(tokens)
        tokens = self.norm2(tokens + ff_out)

        return tokens  # (B, n_summary_tokens, d_model)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)

        # Feedforward
        x = x + self.ff(self.norm2(x))
        return x


class MMRFModel(nn.Module):
    """Markovian Multi-Resolution Forecasting model.

    Uses a sliding window of the w most recent resolutions with full token
    sequences, and compressed summaries for distant resolutions.
    """

    def __init__(
        self,
        resolution_configs: List[Tuple[str, int]],  # [(name, n_assets * n_features), ...]
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        window_width: int = 2,
        summary_dim: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.resolution_names = [rc[0] for rc in resolution_configs]
        self.n_resolutions = len(resolution_configs)
        self.window_width = window_width
        self.d_model = d_model

        # Per-resolution tokenizers
        self.tokenizers = nn.ModuleDict()
        for i, (name, input_dim) in enumerate(resolution_configs):
            self.tokenizers[name] = ResolutionTokenizer(
                input_dim=input_dim, d_model=d_model, resolution_id=i
            )

        # Compressed summary network
        self.summary_net = CompressedSummaryNetwork(
            summary_dim=summary_dim, d_model=d_model, n_summary_tokens=4
        )

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Per-resolution prediction heads
        self.pred_heads = nn.ModuleDict()
        for name in self.resolution_names:
            self.pred_heads[name] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        compressed_summaries: torch.Tensor,  # (B, n_res, summary_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: dict of {resolution_name: (B, seq_len, n_assets * n_features)}
            compressed_summaries: summaries for ALL resolutions

        Returns:
            predictions: dict of {resolution_name: (B,) predicted returns}
            loss: scalar training loss
        """
        B = next(iter(tokens.values())).shape[0]
        device = next(iter(tokens.values())).device

        # Build token sequences for the sliding window (last w resolutions)
        window_tokens = []
        for i, res_name in enumerate(self.resolution_names):
            if i >= self.n_resolutions - self.window_width:
                # Within window: use full token sequence
                tok = self.tokenizers[res_name](tokens[res_name])
                window_tokens.append(tok)

        # Process compressed summaries for distant resolutions
        distant_summaries = compressed_summaries[:, : self.n_resolutions - self.window_width, :]
        summary_tokens = self.summary_net(distant_summaries)  # (B, 4, d_model)

        # Concatenate: [summary_tokens, window_resolution_tokens...]
        all_tokens = [summary_tokens] + window_tokens
        context = torch.cat(all_tokens, dim=1)  # (B, total_len, d_model)

        # Pass through transformer
        h = context
        for block in self.transformer_blocks:
            h = block(h)
        h = self.final_norm(h)

        # Extract predictions: use the last token of each resolution's window
        predictions = {}
        losses = []

        # For summary tokens, use a pooled representation
        summary_repr = h[:, : summary_tokens.shape[1], :].mean(dim=1)  # (B, d_model)

        offset = summary_tokens.shape[1]
        for i, res_name in enumerate(self.resolution_names):
            if i >= self.n_resolutions - self.window_width:
                tok_len = tokens[res_name].shape[1]
                res_repr = h[:, offset : offset + tok_len, :].mean(dim=1)  # (B, d_model)
                offset += tok_len
            else:
                res_repr = summary_repr

            pred = self.pred_heads[res_name](res_repr).squeeze(-1)  # (B,)
            predictions[res_name] = pred

            # Compute loss if target available
            target_key = f"target_{res_name}"
            # Loss will be computed externally

        return {"predictions": predictions, "hidden_states": h}

    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute MSE loss across all resolution predictions."""
        total_loss = torch.tensor(0.0, device=next(iter(output["predictions"].values())).device)
        n = 0
        for res_name in self.resolution_names:
            pred = output["predictions"][res_name]
            target = targets[res_name]
            total_loss = total_loss + F.mse_loss(pred, target)
            n += 1
        return total_loss / max(n, 1)

    def get_memory_estimate(self, batch_size: int = 32, seq_len: int = 30) -> Dict[str, float]:
        """Estimate memory usage for different window widths."""
        d = self.d_model
        n_heads = 8

        estimates = {}
        for w in [1, 2, 3, 4, 5]:
            # Attention memory: O(B * n_heads * L^2 * d)
            n_active = min(w, self.n_resolutions)
            tokens_per_res = seq_len
            total_tokens = 4 + n_active * tokens_per_res  # 4 summary tokens
            attn_mem = batch_size * n_heads * total_tokens ** 2 * 4  # bytes (float32)
            param_mem = sum(p.numel() * 4 for p in self.parameters())
            total_mb = (attn_mem + param_mem) / (1024 ** 2)
            estimates[f"w={w}"] = total_mb

        return estimates


class FullContextARModel(nn.Module):
    """Baseline: full-context autoregressive model using all resolutions."""

    def __init__(
        self,
        resolution_configs: List[Tuple[str, int]],
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.resolution_names = [rc[0] for rc in resolution_configs]
        self.n_resolutions = len(resolution_configs)
        self.d_model = d_model

        self.tokenizers = nn.ModuleDict()
        for i, (name, input_dim) in enumerate(resolution_configs):
            self.tokenizers[name] = ResolutionTokenizer(
                input_dim=input_dim, d_model=d_model, resolution_id=i
            )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.pred_heads = nn.ModuleDict()
        for name in self.resolution_names:
            self.pred_heads[name] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Use ALL resolution tokens
        all_tokens = []
        for res_name in self.resolution_names:
            tok = self.tokenizers[res_name](tokens[res_name])
            all_tokens.append(tok)

        context = torch.cat(all_tokens, dim=1)

        h = context
        for block in self.transformer_blocks:
            h = block(h)
        h = self.final_norm(h)

        predictions = {}
        offset = 0
        for res_name in self.resolution_names:
            tok_len = tokens[res_name].shape[1]
            res_repr = h[:, offset : offset + tok_len, :].mean(dim=1)
            pred = self.pred_heads[res_name](res_repr).squeeze(-1)
            predictions[res_name] = pred
            offset += tok_len

        return {"predictions": predictions, "hidden_states": h}

    def compute_loss(self, output, targets):
        total_loss = torch.tensor(0.0, device=next(iter(output["predictions"].values())).device)
        n = 0
        for res_name in self.resolution_names:
            total_loss = total_loss + F.mse_loss(output["predictions"][res_name], targets[res_name])
            n += 1
        return total_loss / max(n, 1)
