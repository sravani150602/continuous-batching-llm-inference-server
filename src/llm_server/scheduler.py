import heapq
import itertools

from .models import GenerationRequest


class PriorityScheduler:
    def __init__(self, max_batch_size: int, max_batch_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self._waiting: list[tuple[int, int, GenerationRequest]] = []
        self._counter = itertools.count()

    def enqueue(self, request: GenerationRequest) -> None:
        heapq.heappush(self._waiting, (-request.priority, next(self._counter), request))

    def form_batch(self, running: list[GenerationRequest]) -> list[GenerationRequest]:
        batch = sorted(running, key=lambda r: (-r.priority, r.created_at))[: self.max_batch_size]
        token_budget = sum(r.tokens_in_memory for r in batch)
        while self._waiting and len(batch) < self.max_batch_size:
            item = heapq.heappop(self._waiting)
            request = item[2]
            if token_budget + request.tokens_in_memory > self.max_batch_tokens:
                heapq.heappush(self._waiting, item)
                break
            batch.append(request)
            token_budget += request.tokens_in_memory
        return batch

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)
