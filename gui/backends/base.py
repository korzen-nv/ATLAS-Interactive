"""Backend protocol definitions for inference engines.

All propagation backends produce (num_objects+1, H, W) float probability tensors
where channel 0 is background. All click backends produce (1, 1, H, W) float
probability tensors for a single object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable

import torch


@dataclass
class MemoryStatus:
    """Backend-agnostic memory utilization for UI gauges."""
    perm_tokens: int = 0
    work_tokens: int = 0
    max_work_tokens: int = 1
    long_tokens: int = 0
    max_long_tokens: int = 1


@runtime_checkable
class PropagationBackend(Protocol):
    """Protocol for video mask propagation engines (CUTIE, SAM 2, SAM 3)."""

    def step(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        objects: Optional[List[int]] = None,
        *,
        frame_idx: Optional[int] = None,
        idx_mask: bool = True,
        end: bool = False,
        force_permanent: bool = False,
    ) -> torch.Tensor:
        """Process one frame.

        If mask is provided: encode this frame as a user-annotated anchor.
        If mask is None: propagate from memory to segment this frame.

        Args:
            image: (3, H, W) float tensor, values in [0, 1].
            mask: (H, W) index mask or (num_objects, H, W) probs, or None.
            objects: list of object IDs present in the mask.
            frame_idx: absolute video frame index (used by SAM backends
                       for correct frame synchronisation; ignored by CUTIE).
            idx_mask: if True, mask contains object index per pixel.
            end: hint that this is the final frame (skip memory update).
            force_permanent: mark this frame as permanent memory.

        Returns:
            (num_objects+1, H, W) float probability tensor.
        """
        ...

    def clear_memory(self) -> None:
        """Clear all memory (permanent + working + long-term)."""
        ...

    def clear_non_permanent_memory(self) -> None:
        """Clear working + long-term memory, keep permanent."""
        ...

    def clear_sensory_memory(self) -> None:
        """Clear sensory/short-term memory before a propagation run."""
        ...

    def update_config(self, cfg: Any) -> None:
        """Update runtime configuration (mem_every, memory limits, etc.)."""
        ...

    def delete_objects(self, objects: List[int]) -> None:
        """Remove specific objects from tracking."""
        ...

    def output_prob_to_mask(self, output_prob: torch.Tensor) -> torch.Tensor:
        """Convert (num_objects+1, H, W) probs to (H, W) index mask."""
        ...

    def inject_permanent_memory(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        objects: List[int],
    ) -> None:
        """Inject a foreign image+mask pair as a permanent conditioning anchor.

        Unlike ``step()``, this does **not** advance the frame counter, modify
        the current segmentation, or trigger propagation.  The pair is stored
        in permanent memory for cross-video conditioning.

        Args:
            image: (3, H, W) RGB float tensor in [0, 1].
            mask: (H, W) index mask (uint8/int64).
            objects: object IDs present in the mask (excluding background 0).
        """
        ...

    def get_memory_status(self) -> MemoryStatus:
        """Return current memory utilization for UI gauges."""
        ...


@runtime_checkable
class ClickBackend(Protocol):
    """Protocol for single-frame interactive click segmentation."""

    def interact(
        self,
        image: torch.Tensor,
        x: int,
        y: int,
        is_positive: bool,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process a click on the current image.

        Args:
            image: (1, 3, H, W) batched image tensor.
            x, y: click coordinates in image space.
            is_positive: True for foreground, False for background click.
            prev_mask: (1, 1, H, W) previous mask for this object.

        Returns:
            (1, 1, H, W) probability tensor for the clicked object.
        """
        ...

    def unanchor(self) -> None:
        """Reset the current image context."""
        ...

    def undo(self) -> Optional[torch.Tensor]:
        """Undo the last click. Returns updated mask or None."""
        ...
