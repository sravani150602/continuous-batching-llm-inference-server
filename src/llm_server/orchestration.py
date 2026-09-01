import json


class KafkaOrchestrator:
    """Durable request ingress and lifecycle event publishing."""

    def __init__(self, producer, request_topic: str, event_topic: str):
        self.producer = producer
        self.request_topic = request_topic
        self.event_topic = event_topic

    async def publish_request(self, request_id: str, prompt: str, **options) -> None:
        body = json.dumps({"request_id": request_id, "prompt": prompt, **options}).encode()
        await self.producer.send_and_wait(self.request_topic, body, key=request_id.encode())

    async def publish_event(self, request_id: str, event: str, **metadata) -> None:
        body = json.dumps({"request_id": request_id, "event": event, **metadata}).encode()
        await self.producer.send_and_wait(self.event_topic, body, key=request_id.encode())

