"""Streaming generation example — tokens printed in real-time."""

import asyncio
from mymps import MympsClient

async def main():
    client = MympsClient()

    # Make sure a model is loaded first
    client.load_model("microsoft/phi-2")

    print("--- Streaming ---")
    async with client.stream_generate(
        model="microsoft/phi-2",
        prompt="Write a short poem about the sea:\n",
        max_new_tokens=128,
        temperature=0.8,
    ) as stream:
        async for token in stream:
            print(token, end="", flush=True)

    print(f"\n--- Done: {stream.result.tokens_generated} tokens, "
          f"{stream.result.time_s}s, {stream.result.tokens_per_s} tok/s, "
          f"stop={stream.result.stop_reason} ---")

    client.close()

asyncio.run(main())
