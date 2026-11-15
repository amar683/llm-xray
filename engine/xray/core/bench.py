"""Small hardware benchmarks used to put model timings in context."""

from __future__ import annotations

import statistics

import torch

from xray.core.device import now


def memory_bandwidth_gbps(device: torch.device, size_mb: int = 512, repeats: int = 7) -> float:
    """Approximate peak read bandwidth of `device` in GB/s.

    Streams once through a large buffer (a sum over fp16 values). Generating a
    token has to read every weight once, so this bounds how fast decoding can
    possibly go on this machine.
    """
    x = torch.ones(size_mb * 2**20 // 2, dtype=torch.float16, device=device)
    nbytes = x.numel() * x.element_size()
    x.sum()  # warm up
    times = []
    for _ in range(repeats):
        t0 = now(device)
        x.sum()
        times.append(now(device) - t0)
    del x
    return nbytes / statistics.median(times) / 1e9
