import json

from .models import GenerationRequest


class RedisStateStore:
    def __init__(self, redis_client, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl = ttl_seconds

    async def checkpoint(self, request: GenerationRequest) -> None:
        payload = json.dumps(
            {
                "request_id": request.request_id,
                "prompt": request.prompt,
                "generated_token_ids": request.generated_token_ids,
                "priority": request.priority,
                "state": request.state.name,
            }
        )
        await self.redis.set(f"inference:{request.request_id}", payload, ex=self.ttl)

    async def load(self, request_id: str) -> dict | None:
        value = await self.redis.get(f"inference:{request_id}")
        return json.loads(value) if value else None
