"""Shared fake-model helpers for the simulator test suite.

The refactored cost model in ``src.utils.utils`` and ``src.model.model`` reads
``model.cost_constants`` / ``model.config`` keys and calls
``model.dtype_size(key)`` / ``model.kv_size_tokens(tokens)``.  Tests that need a
deterministic, HF-free model use :func:`make_fake_model` so those lookups return
plain numbers instead of ``MagicMock`` objects.
"""

from unittest.mock import MagicMock

from src.model.model import Model


def make_fake_model(*, kv_bytes_per_token: int = 100) -> MagicMock:
    """A ``MagicMock(spec=Model)`` with a complete, deterministic cost model.

    ``kv_size_tokens(t)`` is kept linear (``t * kv_bytes_per_token``) so cache
    capacity behaviour is unchanged from the old ``kv_size_per_token`` API
    (default 100 bytes per token).
    """
    model = MagicMock(spec=Model)
    model.name = "fake"
    model.max_context_size = 10_000_000
    model.config = {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "vocab_size": 1000,
        "head_dim": 64,
        "dtype": "bfloat16",
        "mamba_ssm_dtype": "float32",
        "full_attention_interval": 4,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 64,
        "linear_num_value_heads": 8,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
    }
    model.cost_constants = {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "vocab_size": 1000,
        "head_dim": 64,
        "ld_q": 256,
        "ld_k": 256,
        "ld_v": 512,
        "full_attn_layers": 0.5,
        "linear_attn_layers": 1.5,
        "linear_conv_kernel_dim": 4,
    }
    model.dtype_size = lambda _: 2
    model.kv_size_tokens = lambda tokens: tokens * kv_bytes_per_token
    return model
