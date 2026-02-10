"""Basic example: load a model, run generation, unload."""

from mymps import MympsClient

client = MympsClient()

# Check server health
print("Health:", client.health())

# Load a small model
print("Loading model …")
client.load_model("microsoft/phi-2")

# List loaded models
print("Models:", client.list_models())

# Generate text
result = client.generate(
    model="microsoft/phi-2",
    prompt="The meaning of life is",
    max_new_tokens=64,
    temperature=0.7,
)
print(f"\n--- Generated ({result['tokens_generated']} tokens in {result['time_s']}s) ---")
print(result["text"])

# Cleanup
client.unload_model("microsoft/phi-2")
client.close()
