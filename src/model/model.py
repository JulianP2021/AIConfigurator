from typing import Any, cast

# Third Party
from transformers import AutoConfig, PretrainedConfig

class Model:
    name: str
    config: dict[str, Any]

    def __init__(self, name: str):
        self.name = name
        self.config = fetchArchitecture(self.name)
        print(f"Model {self.name} config: {self.config}")

    @property
    def dtype_size(self) -> float:
        dtype = self.config.get("dtype", "float32")
        if dtype == "float16" or dtype == "bfloat16":
            return 2.0
        elif dtype == "float32":
            return 4.0
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    @property
    def KV_SIZE_PER_TOKEN(self) -> int:
        return self.calculateKVSize(dtype=self.config.get("dtype", "float32"))

    def calculateKVSize(self, dtype: str) -> int:
        if (dtype == 'float32'):
            dtype_size = 4
        elif (dtype == 'float16' or dtype == 'bfloat16'):
            dtype_size = 2
        else:
            dtype_size = 1

        isDeepSeekModel = self.name.startswith("deepseek-ai/DeepSeek-V3") or self.name == "deepseek-ai/DeepSeek-R1";

        isQwen3Model = self.name.lower().startswith("qwen/qwen3-")

        isGLM4Model = self.name.startswith("zai-org/GLM-4.")

        isHunyuanDenseModel = self.name.lower().startswith("tencent/hunyuan-") and self.name.lower() != "tencent/hunyuan-large"

        isHunyuanLargeModel = self.name.lower() == "tencent/hunyuan-large"

        isGQAWithHeadDimModel = isQwen3Model or isGLM4Model or isHunyuanDenseModel;
        tokens = 1
        total_elements = 0
        if (isDeepSeekModel):
            total_elements = self.config["num_hidden_layers"] * tokens * (self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"]);
        elif (isHunyuanLargeModel) :
            cla_share_factor = self.config["cla_share_factor"]
            effective_layers = self.config["num_hidden_layers"] / cla_share_factor
            total_elements = 2 * effective_layers * tokens * self.config["num_key_value_heads"] * self.config["head_size"]
        elif (isGQAWithHeadDimModel):
            total_elements = 2 * self.config["num_hidden_layers"] * tokens * self.config["num_key_value_heads"] * self.config["head_dim"]
        else:
            total_elements = 2 * self.config["num_hidden_layers"] * tokens * self.config["num_key_value_heads"] * self.config["head_size"]

        total_bytes = total_elements * dtype_size
        return total_bytes

def fetchArchitecture(modelName: str) -> dict[str, Any]:
    config = cast(PretrainedConfig, cast(Any, AutoConfig).from_pretrained(modelName, local_files_only = True))
    return config.to_dict()