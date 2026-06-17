class Request:
    isl: int
    osl: int
    prefilled_tokens: int
    decoded_tokens: int
    remaining_prefill_time_ms: int

    def __init__(self, isl: int, osl: int, cached: int):
        self.isl = isl
        self.osl = osl
        self.prefilled_tokens = cached
        self.decoded_tokens = 0
        self.remaining_prefill_time_ms = -1


        assert self.isl >= 0, "ISL must be non-negative"
        assert self.osl >= 0, "OSL must be non-negative"
        assert self.prefilled_tokens < self.isl, "Prefilled tokens must be less than ISL"

    @property
    def remaining_tokens_prefill(self):
        return self.isl - self.prefilled_tokens
    @property
    def remaining_tokens_decode(self):
        return self.osl - self.decoded_tokens