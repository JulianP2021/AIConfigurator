from dataclasses import dataclass

@dataclass
class Hardware:
    name: str
    flops: int
    memoryGB: int
    memoryGB_BW: int

    def __init__(self, name: str, flops: int = 0, memoryGB_BW: int = 0, memoryGB: int = 0):
        if name == "DGX SPARK":
            self.flops = 213*10**12
            self.memoryGB_BW = 273*10**9
            self.memoryGB = 40*10**9