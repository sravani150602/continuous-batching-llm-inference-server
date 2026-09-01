import argparse
import asyncio
import json
import statistics
import time
from llm_server.backend import DeterministicBackend
from llm_server.config import ServerConfig
from llm_server.engine import ContinuousBatchingEngine
from llm_server.models import GenerationRequest


async def run(concurrency: int, requests: int, output_tokens: int, delay_ms: float):
    engine = ContinuousBatchingEngine(
        ServerConfig(max_batch_size=concurrency, kv_pages=4096, scheduler_tick_ms=1),
        DeterministicBackend(delay_ms),
    )
    await engine.start()
    ttfts, latencies = [], []

    async def one(index: int):
        request = GenerationRequest(f"benchmark-{index}", "benchmark prompt", output_tokens)
        started = time.perf_counter()
        first = None
        async for _ in engine.submit(request):
            first = first or time.perf_counter()
        ended = time.perf_counter()
        ttfts.append((first - started) * 1000)
        latencies.append((ended - started) * 1000)

    started = time.perf_counter()
    for offset in range(0, requests, concurrency):
        await asyncio.gather(*(one(i) for i in range(offset, min(offset + concurrency, requests))))
    duration = time.perf_counter() - started
    await engine.close()
    return {
        "requests": requests,
        "concurrency": concurrency,
        "throughput_tokens_per_second": requests * output_tokens / duration,
        "ttft_p50_ms": statistics.median(ttfts),
        "ttft_p99_ms": sorted(ttfts)[max(0, int(len(ttfts) * 0.99) - 1)],
        "latency_p99_ms": sorted(latencies)[max(0, int(len(latencies) * 0.99) - 1)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--step-delay-ms", type=float, default=2)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.concurrency, args.requests, args.output_tokens, args.step_delay_ms)), indent=2))

