import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto


class RequestState(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()
    FAILED = auto()


@dataclass(slots=True)
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    priority: int = 0
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    kv_pages: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    created_at: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    output: asyncio.Queue[int | None] = field(default_factory=asyncio.Queue)

    @property
    def tokens_in_memory(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def finished(self) -> bool:
        return len(self.generated_token_ids) >= self.max_new_tokens
