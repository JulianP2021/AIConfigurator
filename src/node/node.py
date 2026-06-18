from dataclasses import dataclass

from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance

node_id_counter = 0


@dataclass
class Node:
    id: int
    prefill_instances: list[PrefillInstance]
    decode_instances: list[DecodeInstance]

    def __init__(
        self,
        prefill_instances: list[PrefillInstance],
        decode_instances: list[DecodeInstance],
    ):
        global node_id_counter
        self.id = node_id_counter
        node_id_counter += 1
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
