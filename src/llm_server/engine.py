import asyncio
from collections.abc import AsyncIterator

from .backend import DeterministicBackend, ModelBackend
from .config import ServerConfig
from .kv_cache import OutOfPages, PagedKVCache
from .models import GenerationRequest, RequestState
from .scheduler import PriorityScheduler


class ContinuousBatchingEngine:
    def __init__(self, config: ServerConfig | None = None, backend: ModelBackend | None = None):
        self.config = config or ServerConfig()
        self.backend = backend or DeterministicBackend()
        self.cache = PagedKVCache(
            self.config.kv_pages, self.config.kv_page_size, self.config.prefix_cache_capacity
        )
        self.scheduler = PriorityScheduler(self.config.max_batch_size, self.config.max_batch_tokens)
        self.running: list[GenerationRequest] = []
        self.requests: dict[str, GenerationRequest] = {}
        self._task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="continuous-batching-loop")

    async def close(self) -> None:
        self._closed = True
        if self._task:
            await self._task

    async def submit(self, request: GenerationRequest) -> AsyncIterator[str]:
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request id: {request.request_id}")
        request.prompt_token_ids = self.backend.encode(request.prompt)
        self.cache.ensure_capacity(
            request.request_id, request.kv_pages, len(request.prompt_token_ids)
        )
        self.requests[request.request_id] = request
        self.scheduler.enqueue(request)
        while True:
            token = await request.output.get()
            if token is None:
                break
            yield self.backend.decode(token)

    async def _loop(self) -> None:
        while not self._closed or self.running or self.scheduler.waiting_count:
            batch = self.scheduler.form_batch(self.running)
            self.running = batch
            if not batch:
                await asyncio.sleep(self.config.scheduler_tick_ms / 1000)
                continue
            for request in batch:
                request.state = RequestState.RUNNING
            try:
                for request in batch:
                    self.cache.ensure_capacity(
                        request.request_id, request.kv_pages, request.tokens_in_memory + 1
                    )
            except OutOfPages:
                if self.config.preemption_enabled:
                    victim = min(batch, key=lambda r: (r.priority, -r.tokens_in_memory))
                    self.cache.release(victim.request_id, victim.kv_pages)
                    victim.state = RequestState.PREEMPTED
                    self.scheduler.enqueue(victim)
                    self.running.remove(victim)
                    continue
                raise
            next_tokens = await self.backend.step(
                [r.prompt_token_ids + r.generated_token_ids for r in batch]
            )
            survivors = []
            for request, token in zip(batch, next_tokens):
                request.generated_token_ids.append(token)
                if request.first_token_at is None:
                    request.first_token_at = asyncio.get_running_loop().time()
                await request.output.put(token)
                if request.finished:
                    request.state = RequestState.FINISHED
                    self.cache.remember_prefix(request.prompt_token_ids, request.kv_pages)
                    self.cache.release(request.request_id, request.kv_pages)
                    await request.output.put(None)
                else:
                    survivors.append(request)
            self.running = survivors
