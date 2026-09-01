from llm_server.models import GenerationRequest
from llm_server.scheduler import PriorityScheduler


def test_priority_and_batch_limit():
    scheduler = PriorityScheduler(2, 100)
    low = GenerationRequest("low", "x", priority=1, prompt_token_ids=[1])
    high = GenerationRequest("high", "x", priority=9, prompt_token_ids=[1])
    scheduler.enqueue(low)
    scheduler.enqueue(high)
    batch = scheduler.form_batch([])
    assert [request.request_id for request in batch] == ["high", "low"]
