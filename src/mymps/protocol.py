"""Shared constants for client ↔ server communication."""

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555

# REST endpoints
EP_HEALTH = "/health"
EP_MODELS = "/models"
EP_MODELS_LOAD = "/models/load"
EP_INFER = "/infer"
EP_GENERATE = "/generate"

# WebSocket
WS_GENERATE = "/ws/generate"

# Streaming message types
MSG_TOKEN = "token"
MSG_DONE = "done"
MSG_ERROR = "error"
