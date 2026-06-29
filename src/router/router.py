from dataclasses import dataclass

from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import debug_print
from src.request.request import Request


@dataclass
class Router:
    queue: list[Request]
    prefill_instances: list[PrefillInstance]
    decode_instances: list[DecodeInstance]

    def route_requests(self):
        for req in self.queue:
            print(f"Routing request {req.id} with stage {req.stage}")
            # route to prefill or decode instance based on req.stage
            if req.stage == "prefill":
                # find prefill instance with shortest queue
                instance = min(self.prefill_instances, key=lambda x: len(x.queue))
                instance.add_request(req)
            else:
                # route to decode instance with shortest queue
                instance = min(self.decode_instances, key=lambda x: len(x.queue))
                instance.add_request(req)
        self.queue = []

    def log(self):
        debug_print(f"Router state: {len(self.queue)} requests in router queue")
        for i, instance in enumerate(self.prefill_instances):
            debug_print(f"Prefill instance {i} queue length: {len(instance.queue)}")
            instance.log()
        for i, instance in enumerate(self.decode_instances):
            debug_print(f"Decode instance {i} queue length: {len(instance.queue)}")
            instance.log()
