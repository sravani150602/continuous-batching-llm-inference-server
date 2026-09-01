import argparse
import asyncio
import uuid
from .backend import DeterministicBackend, TorchBackend
from .config import ServerConfig
from .engine import ContinuousBatchingEngine
from .models import GenerationRequest


async def demo(args) -> None:
    backend = TorchBackend(args.model, args.device) if args.backend == "torch" else DeterministicBackend(2)
    engine = ContinuousBatchingEngine(ServerConfig.from_env(), backend)
    await engine.start()

    async def generate(prompt: str) -> None:
        request = GenerationRequest(str(uuid.uuid4()), prompt, args.max_tokens)
        print(f"prompt: {prompt}\nresponse: ", end="", flush=True)
        async for token in engine.submit(request):
            print(token, end="", flush=True)
        print()

    await asyncio.gather(*(generate(prompt) for prompt in args.prompts))
    await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["deterministic", "torch"], default="deterministic")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("prompts", nargs="*", default=["Continuous batching", "Paged attention"])
    asyncio.run(demo(parser.parse_args()))


if __name__ == "__main__":
    main()

