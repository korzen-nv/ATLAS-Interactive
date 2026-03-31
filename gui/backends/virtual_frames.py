"""Proxy for frame containers that supports virtual (negative-index) frames
for foreign images injected from global memory."""
from __future__ import annotations

from typing import Dict

import torch


class VirtualFrameProxy:
    """Wraps an image loader/list and adds virtual frames at negative indices.

    Real video frames occupy indices ``[0, N)``.  Virtual frames (from global
    memory) are assigned indices ``-1, -2, …`` so they never collide with real
    frames and sort *before* them in ``_replay_anchors``.

    The proxy is transparent to SAM's ``_get_image_feature`` — it supports
    ``__getitem__`` and ``__len__``, and forwards attribute access
    (``video_height``, ``video_width``, …) to the underlying real container.
    """

    def __init__(self, real_images) -> None:
        self._real = real_images
        self._virtual: Dict[int, torch.Tensor] = {}

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            return self._virtual[index]
        return self._real[index]

    def __len__(self) -> int:
        # Return only real frame count — SAM uses this for propagation bounds.
        return len(self._real)

    def add_virtual(self, image: torch.Tensor) -> int:
        """Register a virtual frame and return its negative index."""
        idx = -(len(self._virtual) + 1)
        self._virtual[idx] = image
        return idx

    def clear_virtual(self) -> None:
        """Remove all virtual frames."""
        self._virtual.clear()

    def __getattr__(self, name):
        return getattr(self._real, name)
