# mymps

MPS compute coprocessor — use a Mac with Apple Silicon as a remote GPU from Linux, connected via SSH tunnel.

Supports: HuggingFace models (any type), TorchScript models, checkpoint + class loading, and ~100 whitelisted torch operations (matmul, softmax, conv2d, SVD, ...).

## Architecture

```
[Linux]  MympsClient  --HTTP-->  [SSH Tunnel :5555]  -->  [Mac]  FastAPI + PyTorch MPS
```

The SSH tunnel handles authentication and encryption — the server itself binds to localhost only.

## Setup

### Linux (client)

```bash
pip install -e .
```

Installs only lightweight client dependencies (httpx, msgpack, numpy). No PyTorch required.

### Mac (server)

```bash
pip install -e '.[server]'

# For HuggingFace model support:
pip install -e '.[server,huggingface]'
```

## Usage

### 1. Start the SSH tunnel (on Linux)

```bash
./scripts/tunnel.sh
```

### 2. Start the server (on Mac)

```bash
python -m mymps.server
```

Verify: `curl localhost:5555/health`

### 3. Torch operations (no model needed)

```python
import numpy as np
from mymps import MympsClient

client = MympsClient()  # connects to localhost:5555

# List available operations
client.list_ops()  # → ["abs", "add", "matmul", "softmax", ...]

# Remote matmul on MPS
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)
result = client.exec("matmul", {"0": A, "1": B})
C = result["output"]  # shape (64, 32)

# Remote softmax
logits = np.random.randn(4, 10).astype(np.float32)
probs = client.exec("softmax", {"input": logits}, dim=-1)

client.close()
```

### 4. Model inference

```python
from mymps import MympsClient

client = MympsClient()

# Load a HuggingFace model (generic AutoModel, not CausalLM)
client.load_model("bert-base-uncased")

# Forward pass with raw tensors
import numpy as np
input_ids = np.array([[101, 2023, 2003, 1037, 3231, 102]], dtype=np.int64)
result = client.infer("bert-base-uncased", {"input_ids": input_ids})

# Load a TorchScript model (file must exist on the Mac)
client.load_model("my-model", model_type="torchscript", model_path="/path/to/model.pt")

# Load a checkpoint with a custom class
client.load_model("my-model", model_type="checkpoint",
                  model_path="/path/to/weights.pt",
                  model_class="mypackage.MyModel")

client.unload_model("bert-base-uncased")
client.close()
```

### 5. Dashboard

```bash
pip install -e '.[dashboard]'
python -m mymps.dashboard
# opens at http://localhost:8080
```

Web UI for monitoring server health, managing models, and running compute operations.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + MPS device info |
| `GET` | `/models` | List loaded models |
| `POST` | `/models/load` | Load a model (huggingface / torchscript / checkpoint) |
| `DELETE` | `/models/{name}` | Unload a model |
| `GET` | `/models/{name}/info` | Model details + metadata |
| `POST` | `/infer` | Forward pass (msgpack tensors, `x-model` header) |
| `GET` | `/ops` | List available torch operations |
| `POST` | `/exec` | Execute a torch op (`x-op` header, `x-kwargs` JSON header, msgpack tensors) |

## Project structure

```
src/mymps/
├── protocol.py          # Shared constants (endpoints)
├── tensor.py            # numpy ↔ msgpack serialization
├── server/
│   ├── app.py           # FastAPI factory + lifespan
│   ├── device.py        # MPS device detection
│   ├── models.py        # Model registry (3 loading strategies)
│   ├── inference.py     # Generic forward pass
│   ├── ops.py           # Whitelisted torch operations
│   └── routes.py        # All HTTP endpoints
├── client/
│   └── client.py        # MympsClient (sync HTTP)
└── dashboard/
    ├── app.py           # Flask app (htmx)
    └── templates/       # Dashboard UI
```
