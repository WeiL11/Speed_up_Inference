"""
Model initialization utilities.

Provides a simple configurable transformer model for benchmarking,
as well as helpers to load pretrained Gemma-3 1B.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configurable transformer (for profiling / benchmarking chapters)
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """Hyperparameters for the simple transformer model."""
    vocab_size: int = 32_000
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.0
    dtype: torch.dtype = torch.float32


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by Gemma / LLaMA)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (will use FlashAttention backend when available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class MLP(nn.Module):
    """Feed-forward network with SiLU gating (Gemma-style)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        intermediate = int(config.hidden_dim * 8 / 3)
        # Round to nearest multiple of 64 for tensor-core alignment
        intermediate = ((intermediate + 63) // 64) * 64

        self.gate_proj = nn.Linear(config.hidden_dim, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """A single transformer block with pre-norm architecture."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = SelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SimpleTransformer(nn.Module):
    """
    A minimal GPT-style transformer for benchmarking.

    Architecture mirrors Gemma/LLaMA: RMSNorm, SiLU-gated MLP, no bias,
    rotary embeddings omitted for simplicity (does not affect profiling).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        count = self.param_count()
        if count >= 1e9:
            return f"{count / 1e9:.2f}B"
        elif count >= 1e6:
            return f"{count / 1e6:.1f}M"
        else:
            return f"{count / 1e3:.1f}K"


def create_model(
    num_layers: int = 6,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    vocab_size: int = 32_000,
    max_seq_len: int = 512,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> SimpleTransformer:
    """Create a SimpleTransformer with the given hyperparameters."""
    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dtype=dtype,
    )
    model = SimpleTransformer(config).to(dtype).to(device)
    return model


def generate_random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int = 32_000,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a random batch of token IDs."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
