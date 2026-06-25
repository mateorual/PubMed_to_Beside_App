"""Optional FAISS vector store helpers for future abstract screening mode."""

from __future__ import annotations

from typing import Any


def build_index(_: list[dict[str, Any]]) -> Any:
    """Reserve the vector-store entry point for post-MVP Mode B."""
    raise NotImplementedError("Vector similarity screening is listed as post-MVP and is not implemented.")
