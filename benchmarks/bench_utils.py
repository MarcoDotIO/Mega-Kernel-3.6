from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BenchResult:
    median_ms: float
    p95_ms: float
    tokens_per_sec: float
    max_memory_mb: float


def cuda_time(
    fn,
    *,
    warmup: int = 10,
    iters: int = 50,
    tokens: int = 1,
    cuda_graph: bool = False,
) -> BenchResult:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        timed_fn = graph.replay
    else:
        timed_fn = fn

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    median = statistics.median(times)
    p95 = sorted(times)[int(0.95 * (len(times) - 1))]
    return BenchResult(
        median_ms=median,
        p95_ms=p95,
        tokens_per_sec=tokens / (median / 1000.0),
        max_memory_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
