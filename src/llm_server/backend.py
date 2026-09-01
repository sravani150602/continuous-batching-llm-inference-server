import asyncio
from abc import ABC, abstractmethod


class ModelBackend(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, token_id: int) -> str: ...

    @abstractmethod
    async def step(self, sequences: list[list[int]]) -> list[int]: ...


class DeterministicBackend(ModelBackend):
    """Fast dependency-free backend used for scheduling tests and benchmarks."""

    def __init__(self, step_delay_ms: float = 0.0):
        self.step_delay_ms = step_delay_ms

    def encode(self, text: str) -> list[int]:
        return [ord(char) % 256 for char in text] or [0]

    def decode(self, token_id: int) -> str:
        return chr(97 + token_id % 26)

    async def step(self, sequences: list[list[int]]) -> list[int]:
        if self.step_delay_ms:
            await asyncio.sleep(self.step_delay_ms / 1000)
        return [(sum(seq[-16:]) + len(seq) * 17) % 256 for seq in sequences]


class TorchBackend(ModelBackend):
    """Hugging Face causal-LM backend; imported lazily so the control plane stays light."""

    def __init__(self, model_name: str = "distilgpt2", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    async def step(self, sequences: list[list[int]]) -> list[int]:
        torch = self.torch
        max_len = max(map(len, sequences))
        padded = [[0] * (max_len - len(seq)) + seq for seq in sequences]
        mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in sequences]
        with torch.inference_mode():
            logits = self.model(
                input_ids=torch.tensor(padded, device=self.device),
                attention_mask=torch.tensor(mask, device=self.device),
            ).logits[:, -1]
        return logits.argmax(-1).cpu().tolist()
