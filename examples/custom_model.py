"""Load a TorchScript (.pt) model from a local file on the Mac."""

import numpy as np

from mymps import MympsClient

client = MympsClient()

# The .pt file must exist on the *Mac* filesystem (the server side).
# For this example, assume you've already exported a model:
#
#   import torch
#   model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU())
#   scripted = torch.jit.script(model)
#   scripted.save("/tmp/simple_model.pt")

MODEL_PATH = "/tmp/simple_model.pt"

print("Loading TorchScript model ...")
client.load_model(
    "my-custom-model",
    model_type="torchscript",
    model_path=MODEL_PATH,
)

print("Models:", client.list_models())
print("Info:", client.model_info("my-custom-model"))

# Run inference — TorchScript models use positional args
x = np.random.randn(4, 10).astype(np.float32)
result = client.infer("my-custom-model", {"0": x})

for key, arr in result.items():
    print(f"  {key}: shape={arr.shape}")

# Cleanup
client.unload_model("my-custom-model")
client.close()
