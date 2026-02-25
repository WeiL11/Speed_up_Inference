"""
Model loader utilities — model-agnostic, Gemma as running example.

Provides:
  - TransformerConfig  : dataclass for all model hyperparameters
  - SimpleTransformer  : configurable GPT-style model (mirrors Gemma architecture)
                         Supports MHA (num_kv_heads == num_heads) and
                         GQA (num_kv_heads < num_heads)
  - create_model()     : build SimpleTransformer from hyperparameters
  - load_hf_model()    : load any HuggingFace model (e.g. google/gemma-3-1b)
  - generate_random_batch() : random token IDs for benchmarking
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """
    Complete hyperparameter set for a decoder-only transformer.

    Designed to match Gemma / LLaMA architecture conventions.
    """
    # Architecture
    vocab_size: int = 32_000
    hidden_dim: int = 1024          # d_model
    num_heads: int = 8              # number of query heads
    num_kv_heads: int = 8           # number of KV heads; if < num_heads → GQA
    intermediate_dim: int = 0       # 0 = auto (8/3 * hidden_dim, rounded to 64)
    num_layers: int = 6
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-6
    tie_weights: bool = True        # tie token embedding ↔ LM head

    # Precision
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
        if self.intermediate_dim == 0:
            # Gemma-style: 8/3 × hidden_dim, rounded up to nearest multiple of 64
            raw = int(self.hidden_dim * 8 / 3)
            self.intermediate_dim = ((raw + 63) // 64) * 64

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def kv_groups(self) -> int:
        """Number of query heads per KV head (GQA grouping factor)."""
        return self.num_heads // self.num_kv_heads


# ---------------------------------------------------------------------------
# Gemma-3 1B preset config
# ---------------------------------------------------------------------------

GEMMA3_1B_CONFIG = TransformerConfig(
    vocab_size=256_000,
    hidden_dim=1152,
    num_heads=4,
    num_kv_heads=1,          # GQA: 1 KV head shared across 4 query heads
    intermediate_dim=6912,
    num_layers=26,
    max_seq_len=8192,
    rms_norm_eps=1e-6,
    tie_weights=True,
)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always compute in float32 for numerical stability, cast back
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SelfAttention(nn.Module):
    """
    Multi-head or Grouped-Query Attention (GQA).

    When num_kv_heads < num_heads, K/V projections are smaller and
    K/V tensors are repeated (expand) to match the number of query heads.
    This matches Gemma-3's GQA design.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_groups = config.kv_groups

        self.q_proj = nn.Linear(config.hidden_dim,
                                config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim,
                                config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim,
                                config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim,
                                config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat K/V so each query head has a corresponding KV head
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Uses FlashAttention backend automatically when available (PyTorch 2.x)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SiLU-gated feed-forward network (Gemma / LLaMA style)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: RMSNorm → Attention → RMSNorm → MLP."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn      = SelfAttention(config)
        self.mlp_norm  = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp       = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SimpleTransformer(nn.Module):
    """
    Configurable decoder-only transformer for benchmarking.

    Architecture: Gemma / LLaMA style
      - RMSNorm (no LayerNorm bias)
      - SiLU-gated MLP (gate_proj + up_proj + down_proj)
      - GQA support (num_kv_heads ≤ num_heads)
      - No biases in linear layers
      - Weight-tied token embedding ↔ LM head (optional)
      - Causal attention via scaled_dot_product_attention(is_causal=True)

    Note: RoPE is omitted for simplicity; positional information does not
    affect profiling/benchmarking results.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config  = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers  = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm    = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        n = self.param_count()
        if n >= 1e9:   return f"{n/1e9:.2f}B"
        if n >= 1e6:   return f"{n/1e6:.1f}M"
        return f"{n/1e3:.1f}K"

    def model_size_mb(self) -> float:
        """Memory footprint of model parameters in MB."""
        bits = {torch.float32: 32, torch.float16: 16, torch.bfloat16: 16}
        b = bits.get(self.config.dtype, 32)
        return self.param_count() * b / 8 / (1024 ** 2)

    def flops_per_token(self) -> int:
        """Approximate FLOPs for a single forward-pass token (Kaplan et al. approximation)."""
        # ~6 * num_params for a transformer without embedding (ties cancel)
        return 6 * self.param_count()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_model(
    num_layers: int = 6,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    intermediate_dim: int = 0,
    vocab_size: int = 32_000,
    max_seq_len: int = 2048,
    tie_weights: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> SimpleTransformer:
    """
    Build a SimpleTransformer from hyperparameters.

    Args:
        num_kv_heads: KV attention heads. None → same as num_heads (MHA).
                      Set < num_heads for GQA (e.g. Gemma-3 uses 1 KV head).
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads

    config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        tie_weights=tie_weights,
        dtype=dtype,
    )
    model = SimpleTransformer(config).to(dtype).to(device)
    return model


def load_hf_model(
    model_name: str = "google/gemma-3-1b",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **kwargs,
):
    """
    Load a pretrained model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model ID (e.g. "google/gemma-3-1b").
        dtype: Model dtype. bfloat16 recommended for Gemma on A100/H100.
        device: Target device.
        **kwargs: Passed to AutoModelForCausalLM.from_pretrained().

    Returns:
        (model, tokenizer)

    Example:
        model, tokenizer = load_hf_model("google/gemma-3-1b")
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")

    print(f"Loading {model_name} from HuggingFace Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        **kwargs,
    )
    model.eval()
    print(f"  Loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


def generate_random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int = 32_000,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a random batch of token IDs for benchmarking."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
