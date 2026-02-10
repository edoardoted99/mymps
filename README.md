# mymps

Remote MPS inference server — offload ML/AI workloads from a Linux machine to a Mac with Apple Silicon, connected via SSH tunnel.

## Architecture

```
[Linux]  MympsClient  --HTTP/WS-->  [SSH Tunnel :5555]  -->  [Mac]  FastAPI + PyTorch MPS
```

The SSH tunnel handles authentication and encryption — the server itself binds to localhost only, no extra auth needed.

## Setup

### Linux (client)

```bash
pip install -e .
```

This installs only the lightweight client dependencies (httpx, websockets, msgpack, numpy). No PyTorch required.

### Mac (server)

```bash
# Option A: deploy from Linux via rsync
./scripts/deploy.sh

# Option B: manually on the Mac
pip install -e '.[server]'
```

The `server` extra pulls in torch, transformers, accelerate, diffusers, uvicorn, and fastapi.

## Usage

### 1. Start the SSH tunnel (on Linux)

```bash
./scripts/tunnel.sh
```

This opens `localhost:5555` on Linux → forwards to port 5555 on the Mac via `ssh -p 26996 edoardo.tedesco@openport.io`.

Configure with env vars: `MYMPS_USER`, `MYMPS_HOST`, `MYMPS_SSH_PORT`, `MYMPS_PORT`.

### 2. Start the server (on Mac)

```bash
./scripts/start-server.sh
# or directly:
python -m mymps.server
```

Verify with: `curl localhost:5555/health`

### 3. Use the client (on Linux)

```python
from mymps import MympsClient

client = MympsClient()  # connects to localhost:5555

# check server
client.health()

# load a model onto MPS
client.load_model("microsoft/phi-2")

# generate text
result = client.generate("microsoft/phi-2", "The meaning of life is", max_new_tokens=64)
print(result["text"])

# streaming generation
import asyncio

async def stream():
    async with client.stream_generate("microsoft/phi-2", "Write a poem:\n") as s:
        async for token in s:
            print(token, end="", flush=True)

asyncio.run(stream())

# cleanup
client.unload_model("microsoft/phi-2")
client.close()
```

### 4. Dashboard (on Linux)

```bash
pip install -e '.[dashboard]'
python -m mymps.dashboard
# opens at http://localhost:8080
```

A minimal web UI (Flask + htmx) that auto-refreshes every 5s showing server health, loaded models, and lets you load/unload models and run generation — all through the tunnel.

Configure with: `MYMPS_DASH_HOST`, `MYMPS_DASH_PORT`, `MYMPS_SERVER_HOST`, `MYMPS_SERVER_PORT`.

## API

### REST

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + MPS GPU info |
| `GET` | `/models` | List loaded models |
| `POST` | `/models/load` | Load a HuggingFace model onto MPS |
| `DELETE` | `/models/{name}` | Unload a model |
| `POST` | `/infer` | Raw tensor inference (msgpack binary) |
| `POST` | `/generate` | Text generation (non-streaming) |

### WebSocket

`/ws/generate` — client sends `{model, prompt, max_new_tokens, temperature, top_p}`, server streams back `{type: "token", text: "..."}` per token, then `{type: "done", ...}` with stats.

## Project structure

```
src/mymps/
├── protocol.py          # Shared constants (endpoints, message types)
├── tensor.py            # numpy <-> msgpack serialization
├── server/
│   ├── app.py           # FastAPI factory + lifespan
│   ├── device.py        # MPS device detection
│   ├── models.py        # Model registry (load/unload/list)
│   ├── inference.py     # Generic forward pass
│   ├── streaming.py     # Token-by-token WebSocket generation
│   └── routes.py        # All HTTP + WS endpoints
├── client/
│   ├── client.py        # MympsClient (sync HTTP + async WS)
│   └── stream.py        # TokenStream async iterator
└── dashboard/
    ├── app.py           # Flask app (htmx partials)
    └── templates/       # Single-page dark UI
```
