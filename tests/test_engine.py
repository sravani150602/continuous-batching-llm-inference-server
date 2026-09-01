import asyncio
from llm_server.backend import DeterministicBackend
from llm_server.config import ServerConfig
from llm_server.engine import ContinuousBatchingEngine
from llm_server.models import GenerationRequest, RequestState


async def test_concurrent_streaming_requests_finish():
    config = ServerConfig(max_batch_size=4, kv_pages=64, scheduler_tick_ms=1)
    engine = ContinuousBatchingEngine(config, DeterministicBackend())
    await engine.start()

    async def consume(index):
        request = GenerationRequest(f"r{index}", f"prompt {index}", max_new_tokens=5)
        tokens = [token async for token in engine.submit(request)]
        return request, tokens

    results = await asyncio.gather(*(consume(i) for i in range(6)))
    await engine.close()
    assert all(len(tokens) == 5 for _, tokens in results)
    assert all(request.state is RequestState.FINISHED for request, _ in results)
    assert engine.cache.free_pages == config.kv_pages

