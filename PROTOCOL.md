# mymps — Protocol Prompt for AI Agents

You have access to a remote MPS compute coprocessor running on a Mac with Apple Silicon, reachable via `mymps`. It acts as a remote GPU: you send tensors and operations, it executes them on the MPS device and returns results.

## Setup

```python
from mymps import MympsClient
import numpy as np

client = MympsClient()  # default: localhost:5555
```

## What you can do

### 1. Execute torch operations remotely (no model needed)

Use `client.exec(op, inputs, **kwargs)` to run any whitelisted torch operation on MPS.

- `op`: operation name (string), e.g. `"matmul"`, `"softmax"`, `"conv2d"`, `"svd"`
- `inputs`: dict of numpy arrays. Use `"0"`, `"1"`, `"2"` as keys for positional args, or named keys for keyword args
- `**kwargs`: scalar keyword arguments (e.g. `dim=-1`)
- Returns: `dict[str, np.ndarray]` — output tensors as numpy arrays

```python
# Matrix multiplication
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)
result = client.exec("matmul", {"0": A, "1": B})
C = result["output"]  # shape (64, 32)

# Softmax with dim kwarg
logits = np.random.randn(4, 10).astype(np.float32)
probs = client.exec("softmax", {"input": logits}, dim=-1)
P = probs["output"]  # shape (4, 10), rows sum to 1

# SVD — returns multiple tensors
M = np.random.randn(8, 5).astype(np.float32)
svd = client.exec("svd", {"0": M})
U, S, Vh = svd["0"], svd["1"], svd["2"]

# Conv2d — input (N,C,H,W) + weight (C_out,C_in,kH,kW)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
w = np.random.randn(16, 3, 3, 3).astype(np.float32)
out = client.exec("conv2d", {"input": x, "weight": w})
```

### 2. List available operations

```python
ops = client.list_ops()
# Returns sorted list: ["abs", "add", "adaptive_avg_pool2d", "amax", "amin", "argmax", ...]
```

Available categories (~100 ops):
- **Element-wise**: abs, neg, exp, log, sqrt, sin, cos, tanh, sigmoid, add, sub, mul, div, clamp, pow, maximum, minimum, ...
- **Reductions**: sum, mean, prod, amax, amin, argmax, argmin, norm
- **Linear algebra**: matmul, mm, bmm, mv, dot, einsum, transpose, svd, eig, inv, det, solve, cholesky, qr
- **Activations**: relu, gelu, silu, softmax, log_softmax, layer_norm
- **Conv/Pool**: conv1d, conv2d, conv3d, avg_pool2d, max_pool2d, adaptive_avg_pool2d
- **Shape**: reshape, flatten, squeeze, unsqueeze, cat, stack, split, chunk, permute
- **FFT**: fft, ifft, rfft, irfft
- **Comparison**: eq, ne, gt, lt, ge, le, where
- **Other**: sort, topk, cumsum, cumprod, zeros_like, ones_like, rand_like

### 3. Load and run models

Three loading strategies:

```python
# HuggingFace (any AutoModel — not just CausalLM)
client.load_model("bert-base-uncased", model_type="huggingface")

# TorchScript (.pt file on the Mac)
client.load_model("my-model", model_type="torchscript", model_path="/path/to/model.pt")

# Checkpoint (state_dict + Python class)
client.load_model("my-model", model_type="checkpoint",
                  model_path="/path/to/weights.pt",
                  model_class="mypackage.MyModel")
```

Run inference:

```python
# Keyword args (HuggingFace style)
result = client.infer("bert-base-uncased", {
    "input_ids": np.array([[101, 2023, 2003, 102]], dtype=np.int64),
    "attention_mask": np.ones((1, 4), dtype=np.int64),
})
# result: dict[str, np.ndarray] — e.g. {"last_hidden_state": array(...)}

# Positional args (TorchScript style)
result = client.infer("my-model", {"0": np.random.randn(4, 10).astype(np.float32)})
```

### 4. Other methods

```python
client.health()              # server status, device info, models loaded
client.stats()               # CPU %, memory, MPS GPU memory, load avg
client.list_models()         # list loaded models
client.model_info("name")    # model details + metadata
client.unload_model("name")  # free memory
client.close()               # close connection
```

## Important rules

- All tensors are `np.ndarray`. Use `np.float32` for most operations. Use `np.int64` for indices/token IDs.
- Positional args use string keys: `"0"`, `"1"`, `"2"`, ...
- Named args use their actual name: `"input"`, `"weight"`, etc.
- Scalar kwargs (like `dim=-1`) go as Python kwargs to `exec()`, not in the inputs dict.
- The server runs on MPS (Apple Silicon GPU). Some operations may fall back to CPU automatically.
- Models must be loaded before inference. Use `unload_model()` to free memory.
- HuggingFace models require the `[huggingface]` extra installed on the server.
- TorchScript/checkpoint files must exist on the **Mac filesystem**, not on the client machine.

## Connection

Default: `localhost:5555` via SSH tunnel. The tunnel must be running for the client to reach the Mac.

```python
# Custom host/port
client = MympsClient(host="10.0.0.5", port=5555, timeout=300.0)

# Context manager
with MympsClient() as client:
    result = client.exec("matmul", {"0": A, "1": B})
```
