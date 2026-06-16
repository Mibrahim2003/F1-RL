"""Local FastAPI/WebSocket backend: the Python simulation engine is the single source
of truth; the browser renders state and sends input. Nothing here imports a renderer."""

from f1rl.server.app import create_app

__all__ = ["create_app"]
