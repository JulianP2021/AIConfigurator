from dataclasses import dataclass

from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import debug_print
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
                instance.add_request(req)
            else:
                # route to decode instance with shortest queue
                instance = min(self.decode_instnces, key=lambda x: len(x.queue))
                instance.add_request(req)
        self.queue = []

    def finish_requests(self):
        self.route_requests()
        total_time_ms = 0
        finished_requests: list[Request] = []
        prefilled_requests: list[Request] = []

        time_to_next_completion = min([
            instance.time_to_next_completion()
            for instance in self.prefill_instances
            if instance.queue or instance.upload_queue
        ])

        while any(instance.queue for instance in self.prefill_instances) or any(
            instance.upload_queue for instance in self.prefill_instances
        ):
            assert time_to_next_completion >= 0, (
                "Time to next completion should be non-negative"
            )
            for instance in self.prefill_instances:
                debug_print(
                    f"Processing prefill instance with queue length {len(instance.queue)}, upload queue length {len(instance.upload_queue)}"
                )
                prefilled_requests_instance = instance.process_queue(
                    time_to_next_completion
                )
                prefilled_requests.extend(prefilled_requests_instance)
                debug_print(
                    f"Finished processing prefill instance, {len(prefilled_requests_instance)} requests finished, total time passed: {total_time_ms / 1000} seconds"
                )
            for instance in self.decode_instnces:
                debug_print(
                    f"Processing decode instance with queue length {len(instance.queue)}, download queue length {len(instance.download_queue)}"
                )
                decoded_requests = instance.process_queue(time_to_next_completion)
                finished_requests.extend(decoded_requests)
            total_time_ms += time_to_next_completion
            self.queue.extend(prefilled_requests)
            prefilled_requests = []

            times = [
                instance.time_to_next_completion()
                for instance in self.prefill_instances
                if instance.queue or instance.upload_queue
            ]
            debug_print(
                f"Times: {times}, self.prefill_instances: {[len(instance.queue) for instance in self.prefill_instances]}, upload_queues: {[len(instance.upload_queue) for instance in self.prefill_instances]}, self.queue: {len(self.queue)}"
            )
            self.route_requests()

            if all(
                not instance.queue and not instance.upload_queue
                for instance in self.prefill_instances
            ):
                break
            self.log()
            time_to_next_completion = min(times)

        # empty prefill queues, then finish decode queues
        assert not any(instance.queue for instance in self.prefill_instances)
        assert not any(instance.upload_queue for instance in self.prefill_instances), (
            "All prefill queues should be empty"
        )

        decode_max_time_ms = 0
        for instance in self.decode_instnces:
            decode_time, finished = instance.finish_queue()
            decode_max_time_ms = max(decode_max_time_ms, decode_time)
            finished_requests.extend(finished)
        total_time_ms += decode_max_time_ms
        return total_time_ms, finished_requests

    def log(self):
        debug_print(f"Router state: {len(self.queue)} requests in router queue")
        for i, instance in enumerate(self.prefill_instances):
            debug_print(f"Prefill instance {i} queue length: {len(instance.queue)}")
            instance.log()
        for i, instance in enumerate(self.decode_instnces):
            debug_print(f"Decode instance {i} queue length: {len(instance.queue)}")
            instance.log()
