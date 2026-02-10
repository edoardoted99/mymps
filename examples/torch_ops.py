"""Execute remote torch operations on the MPS coprocessor."""

import numpy as np

from mymps import MympsClient

client = MympsClient()

# List available operations
ops = client.list_ops()
print(f"{len(ops)} operations available: {ops[:10]} ...")

# ── matmul ──────────────────────────────────────────────────────────
A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 32).astype(np.float32)

result = client.exec("matmul", {"0": A, "1": B})
C = result["output"]
print(f"matmul: ({A.shape}) @ ({B.shape}) → {C.shape}")

# ── softmax ─────────────────────────────────────────────────────────
logits = np.random.randn(4, 10).astype(np.float32)
probs = client.exec("softmax", {"input": logits}, dim=-1)
print(f"softmax: {logits.shape} → {probs['output'].shape}, row sums ≈ {probs['output'].sum(axis=-1)}")

# ── conv2d ──────────────────────────────────────────────────────────
# input: (N, C_in, H, W), weight: (C_out, C_in, kH, kW)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
w = np.random.randn(16, 3, 3, 3).astype(np.float32)
conv_out = client.exec("conv2d", {"input": x, "weight": w})
print(f"conv2d: input={x.shape}, weight={w.shape} → {conv_out['output'].shape}")

# ── svd ─────────────────────────────────────────────────────────────
M = np.random.randn(8, 5).astype(np.float32)
svd_result = client.exec("svd", {"0": M})
for k, v in svd_result.items():
    print(f"svd [{k}]: {v.shape}")

client.close()
