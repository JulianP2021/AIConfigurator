from dataclasses import dataclass

from src.instances.prefill import PrefillInstance
from src.instances.decode import DecodeInstance
from src.request.request import Request

@dataclass
class Router:
    queue: list[Request]
    prefill_instances: list[PrefillInstance]
    decode_instnces: list[DecodeInstance]

    def route_requests(self):
        for req in self.queue:
            # route to prefill or decode instance based on req.stage
            if req.stage == "prefill":
                # find prefill instance with shortest queue
                instance = min(self.prefill_instances, key=lambda x: len(x.queue))
                instance.queue.append(req)
            else:
                # route to decode instance with shortest queue
                instance = min(self.decode_instnces, key=lambda x: len(x.queue))
                instance.queue.append(req)
        self.queue = []

    def finish_requests(self):
        self.route_requests()
        total_time_ms = 0
        finished_requests: list[Request] = []
        prefilled_requests: list[Request] = []
        for instance in self.prefill_instances:
            prefill_time, finished = instance.finish_queue()
            total_time_ms += prefill_time
            prefilled_requests.extend(finished)
        self.queue.extend(prefilled_requests)
        for req in self.queue:
            print(f"Routing request with id: {req.id} from {req.stage} to decode after {total_time_ms / 1000} seconds")
        self.route_requests()

        ## requests could be batched, that are not in decode yet
        for instance in self.decode_instnces:
            decode_time, finished = instance.finish_queue()
            total_time_ms += decode_time
            finished_requests.extend(finished)
        return total_time_ms, finished_requests