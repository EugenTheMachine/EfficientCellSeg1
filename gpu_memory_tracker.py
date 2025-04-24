from typing import Dict

import torch
from torch.cuda import max_memory_allocated, max_memory_reserved, reset_peak_memory_stats


class GPUMemoryTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        return {"peak_allocated": max_memory_allocated() / 1024**2, "peak_reserved": max_memory_reserved() / 1024**2}
