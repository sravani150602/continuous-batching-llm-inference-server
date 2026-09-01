import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerConfig:
    max_batch_size: int = 32
    max_batch_tokens: int = 2048
    kv_pages: int = 1024
    kv_page_size: int = 16
    scheduler_tick_ms: int = 5
    preemption_enabled: bool = True
    prefix_cache_capacity: int = 256
    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap_servers: str = "localhost:9092"
    request_topic: str = "inference.requests"
    event_topic: str = "inference.events"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        def number(name: str, default: int) -> int:
            return int(os.getenv(name, default))

        return cls(
            max_batch_size=number("MAX_BATCH_SIZE", 32),
            max_batch_tokens=number("MAX_BATCH_TOKENS", 2048),
            kv_pages=number("KV_PAGES", 1024),
            kv_page_size=number("KV_PAGE_SIZE", 16),
            scheduler_tick_ms=number("SCHEDULER_TICK_MS", 5),
            preemption_enabled=os.getenv("PREEMPTION_ENABLED", "true").lower() == "true",
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        )
