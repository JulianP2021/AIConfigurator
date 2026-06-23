from src.hardware.hardware import Hardware
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.model.model import Model


node_id_counter = 0


class Node:
    hardware: Hardware
    id: int
    prefill_instances: list[PrefillInstance]
    decode_instances: list[DecodeInstance]

    def __init__(
        self,
        hardware: Hardware,
        batch_size: int = 10,
        model_name: str = "Qwen/Qwen3-8B",
        prefill_instances: int = 0,
        decode_instances: int = 0,
    ):
        global node_id_counter
        self.id = node_id_counter
        node_id_counter += 1
        self.hardware = hardware

        assert prefill_instances + decode_instances > 0, (
            "At least one instance must be greater than 0"
        )
        assert prefill_instances + decode_instances == hardware.spec.num_gpus, (
            "Total instances must be equal to number of GPUs"
        )

        self.prefill_instances = [
            PrefillInstance(
                hardware=hardware.spec.gpu_hardware, model=Model(model_name)
            )
            for _ in range(prefill_instances)
        ]

        self.decode_instances = [
            DecodeInstance(
                hardware=hardware.spec.gpu_hardware,
                max_batch_size=batch_size,
                model=Model(model_name),
            )
            for _ in range(decode_instances)
        ]
