"""Load a generic HuggingFace model and run a forward pass with raw tensors."""

import numpy as np

from mymps import MympsClient

client = MympsClient()

# Check server health
print("Health:", client.health())

# Load a generic encoder model (not CausalLM!)
print("Loading model ...")
client.load_model("bert-base-uncased", model_type="huggingface")

# List loaded models
print("Models:", client.list_models())

# Model info
print("Info:", client.model_info("bert-base-uncased"))

# Prepare dummy input — BERT expects (batch, seq_len) int64 input_ids
input_ids = np.array([[101, 2023, 2003, 1037, 3231, 102]], dtype=np.int64)
attention_mask = np.ones_like(input_ids, dtype=np.int64)

result = client.infer("bert-base-uncased", {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
})

for key, arr in result.items():
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

# Cleanup
client.unload_model("bert-base-uncased")
client.close()
