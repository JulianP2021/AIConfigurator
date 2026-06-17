from typing import Any, cast

# Third Party
from transformers import AutoConfig, PretrainedConfig

class Model:
    name: str
    config: dict[str, Any]

    def __init__(self, name: str):
        self.name = name
        self.config = fetchArchitecture(self.name)


def fetchArchitecture(modelName: str):
    config = cast(PretrainedConfig, cast(Any, AutoConfig).from_pretrained(modelName))
    config.to_dict()
    return config.to_dict()