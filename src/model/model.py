from functools import cached_property
from typing import Any, cast

# Third Party
from transformers import AutoConfig, PretrainedConfig


class Model:
    name: str
    config: dict[str, Any]

    def __init__(self, name: str):
        self.name = name
        self.config = fetch_architecture(self.name)
        # print(f"Model {self.name} config: {self.config}")

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

    @cached_property
    def dtype_size(self) -> float:
        dtype = self.config.get("dtype", "float32")
        if dtype == "float16" or dtype == "bfloat16":
            return 2.0
        if dtype == "float32":
            return 4.0
        raise ValueError(f"Unsupported data type: {dtype}")

    @cached_property
    def kv_size_per_token(self) -> int:
        # KV Calculation based on lmcache kv calculator:
        dtype = self.config.get("dtype", "float32")
        if dtype == "float32":
            dtype_size = 4
        elif dtype == "float16" or dtype == "bfloat16":
            dtype_size = 2
        else:
            dtype_size = 1

        is_deep_seek_model = (
            self.name.startswith("deepseek-ai/DeepSeek-V3")
            or self.name == "deepseek-ai/DeepSeek-R1"
        )
        is_qwen3_model = self.name.lower().startswith("qwen/qwen3-")

        is_glm4_model = self.name.startswith("zai-org/GLM-4.")

        is_hunyuan_dense_model = (
            self.name.lower().startswith("tencent/hunyuan-")
            and self.name.lower() != "tencent/hunyuan-large"
        )

        is_hunyuan_large_model = self.name.lower() == "tencent/hunyuan-large"

        is_gqa_with_head_dim_model = (
            is_qwen3_model or is_glm4_model or is_hunyuan_dense_model
        )
        tokens = 1
        total_elements = 0
        if is_deep_seek_model:
            total_elements = (
                self.config["num_hidden_layers"]
                * tokens
                * (self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"])
            )
        elif is_hunyuan_large_model:
            cla_share_factor = self.config["cla_share_factor"]
            effective_layers = self.config["num_hidden_layers"] / cla_share_factor
            total_elements = (
                2
                * effective_layers
                * tokens
                * self.config["num_key_value_heads"]
                * self.config["head_size"]
            )
        elif is_gqa_with_head_dim_model:
            total_elements = (
                2
                * self.config["num_hidden_layers"]
                * tokens
                * self.config["num_key_value_heads"]
                * self.config["head_dim"]
            )
        else:
            total_elements = (
                2
                * self.config["num_hidden_layers"]
                * tokens
                * self.config["num_key_value_heads"]
                * self.config["head_size"]
            )

        return total_elements * dtype_size

    @cached_property
    def cost_constants(self) -> dict[str, float | int]:
        cfg = self.config

        d_kv = cfg.get(
            "head_dim",
            cfg["hidden_size"] // cfg["num_attention_heads"],
        )

        return {
            "hidden_size": int(cfg["hidden_size"]),
            "intermediate_size": int(cfg["intermediate_size"]),
            "num_hidden_layers": int(cfg["num_hidden_layers"]),
            "num_key_value_heads": int(cfg["num_key_value_heads"]),
            "vocab_size": int(cfg["vocab_size"]),
            "d_kv": int(d_kv),
            "dtype_size": float(self.dtype_size),
            # FLOPs
            "output_flops": int(2 * cfg["hidden_size"] * cfg["vocab_size"]),
            # Memory
            "matrices": int(
                2 * self.dtype_size * cfg["hidden_size"] ** 2
                + 3 * self.dtype_size * cfg["intermediate_size"] * cfg["hidden_size"]
            ),
            "embedding_memory": int(
                2 * self.dtype_size * cfg["hidden_size"] * cfg["vocab_size"]
            ),
        }


def fetch_architecture(model_name: str) -> dict[str, Any]:
    config = cast(
        PretrainedConfig,
        cast(Any, AutoConfig).from_pretrained(model_name, local_files_only=True),
    )
    return config.to_dict()
