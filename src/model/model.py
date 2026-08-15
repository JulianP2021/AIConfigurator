from functools import cached_property, lru_cache
from typing import Any, ClassVar, Self, cast

# Third Party
from transformers import AutoConfig, PretrainedConfig


class Model:
    name: str
    config: dict[str, Any]

    _instances: ClassVar[dict[str, Self]] = {}

    def __new__(cls, name: str | None = None) -> Self:
        if name is None:
            return super().__new__(cls)
        instance = cls._instances.get(name)
        if instance is None:
            instance = super().__new__(cls)
            instance.name = name
            cls._instances[name] = instance
        return instance

    def __init__(self, name: str):
        if not hasattr(self, "config"):
            config = fetch_architecture(name)
            self.config = config.get("text_config") or config

    @cached_property
    def max_context_size(self) -> int:
        for key in (
            "max_position_embeddings",
            "max_sequence_length",
            "n_positions",
            "seq_length",
        ):
            value = self.config.get(key)
            if value is not None:
                return int(value)
        raise ValueError(
            f"Cannot determine context size for model {self.name} from config "
            "(looked for max_position_embeddings, max_sequence_length, "
            "n_positions, seq_length)"
        )

    @lru_cache(maxsize=100)  # noqa: B019
    def dtype_size(self, key: str) -> int:
        dtype = self.config.get(key, "float32")
        if dtype == "float16" or dtype == "bfloat16":
            return 2
        if dtype == "float32":
            return 4
        if dtype == "float8":
            return 1
        raise ValueError(f"Unsupported data type: {dtype}")

    @lru_cache(maxsize=1000)  # noqa: B019
    def kv_size_tokens(self, tokens: int) -> int:
        cfg = self.config
        cc = self.cost_constants
        size = 0

        linear_attn_layers = cc["linear_attn_layers"]

        total_elements = (
            2
            * cc["full_attn_layers"]
            * tokens
            * int(cfg["num_key_value_heads"])
            * int(cfg["head_dim"])
        )
        size += total_elements * self.dtype_size("dtype")

        state_matrix = 1 * cc["ld_v"] * cc["head_dim"] * linear_attn_layers

        conv_matrix = (
            1
            * (2 * cc["ld_k"] + cc["ld_v"])
            * cc["linear_conv_kernel_dim"]
            * linear_attn_layers
        )
        size += state_matrix * self.dtype_size(
            "mamba_ssm_dtype"
        ) + conv_matrix * self.dtype_size("dtype")
        return int(size)

    @cached_property
    def cost_constants(self) -> dict[str, float | int]:
        c = self.config

        full_attn_layers = int(c.get("num_hidden_layers", 0)) / int(
            c.get("full_attention_interval", 4)
        )

        return {
            "hidden_size": int(c["hidden_size"]),
            "intermediate_size": int(c["intermediate_size"]),
            "num_hidden_layers": int(c["num_hidden_layers"]),
            "num_key_value_heads": int(c["num_key_value_heads"]),
            "vocab_size": int(c["vocab_size"]),
            "head_dim": int(c.get("head_dim", 0)),
            "ld_q": int(c.get("linear_num_key_heads", 0))
            * int(c.get("linear_key_head_dim", 0)),
            "ld_k": int(c.get("linear_num_key_heads", 0))
            * int(c.get("linear_key_head_dim", 0)),
            "ld_v": int(c.get("linear_num_value_heads", 0))
            * int(c.get("linear_value_head_dim", 0)),
            "full_attn_layers": full_attn_layers,
            "linear_attn_layers": int(c.get("num_hidden_layers", 0)) - full_attn_layers,
            "linear_conv_kernel_dim": int(c.get("linear_conv_kernel_dim", 0)),
        }


def fetch_architecture(model_name: str) -> dict[str, Any]:
    config = cast(
        PretrainedConfig,
        cast(Any, AutoConfig).from_pretrained(model_name, local_files_only=True),
    )
    return config.to_dict()
