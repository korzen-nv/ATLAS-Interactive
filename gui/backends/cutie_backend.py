"""CUTIE + RITM backend — thin wrapper over existing InferenceCore and ClickController."""
from __future__ import annotations

from typing import Any, List, Optional

import torch
from omegaconf import DictConfig

from gui.backends.base import MemoryStatus


class CutieBackend:
    """Wraps InferenceCore to satisfy the PropagationBackend protocol."""

    def __init__(self, cutie_model, cfg: DictConfig) -> None:
        from gui.cutie.inference.inference_core import InferenceCore
        self._core = InferenceCore(cutie_model, cfg)
        self._cfg = cfg

    # -- PropagationBackend interface ------------------------------------------

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
        return self._core.step(
            image, mask, objects,
            idx_mask=idx_mask, end=end, force_permanent=force_permanent,
        )

    def clear_memory(self) -> None:
        self._core.clear_memory()

    def clear_non_permanent_memory(self) -> None:
        self._core.clear_non_permanent_memory()

    def clear_sensory_memory(self) -> None:
        self._core.clear_sensory_memory()

    def update_config(self, cfg: Any) -> None:
        self._core.update_config(cfg)

    def delete_objects(self, objects: List[int]) -> None:
        self._core.delete_objects(objects)

    def output_prob_to_mask(self, output_prob: torch.Tensor) -> torch.Tensor:
        return self._core.output_prob_to_mask(output_prob)

    def get_memory_status(self) -> MemoryStatus:
        try:
            return MemoryStatus(
                perm_tokens=self._core.memory.work_mem.perm_size(0),
                work_tokens=self._core.memory.work_mem.non_perm_size(0),
                max_work_tokens=self._core.memory.max_work_tokens,
                long_tokens=self._core.memory.long_mem.non_perm_size(0),
                max_long_tokens=self._core.memory.max_long_tokens,
            )
        except AttributeError:
            return MemoryStatus()

    # -- Expose internals that controllers still read for spinbox defaults -----

    @property
    def memory(self):
        """Direct access for CUTIE-specific memory tuning UI."""
        return self._core.memory

    @property
    def mem_every(self):
        return self._core.mem_every


class RitmClickBackend:
    """Wraps ClickController to satisfy the ClickBackend protocol."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda') -> None:
        from gui.click_controller import ClickController
        self._ctrl = ClickController(checkpoint_path, device=device)

    def interact(
        self,
        image: torch.Tensor,
        x: int,
        y: int,
        is_positive: bool,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._ctrl.interact(image, x, y, is_positive, prev_mask)

    def unanchor(self) -> None:
        self._ctrl.unanchor()

    def undo(self) -> Optional[torch.Tensor]:
        return self._ctrl.undo()
