"""SAM 3 backend — standalone implementation for the SAM 3 video model.

Unlike SAM 2, SAM 3 has a fundamentally different API:
- ``init_state`` takes ``resource_path`` (not ``video_path``)
- No ``add_new_mask``; uses ``add_prompt`` with points/text/boxes
- ``propagate_in_video`` yields ``(frame_idx, outputs_dict)`` not tuples
- Has native text-prompt support via a built-in BPE tokenizer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from gui.backends.base import MemoryStatus

log = logging.getLogger(__name__)


# ── Propagation ──────────────────────────────────────────────────────────────


class Sam3PropagationBackend:
    """Wraps the SAM 3 video model for mask propagation.

    Uses the lower-level ``Sam3VideoInferenceWithInstanceInteractivity``
    model directly (via ``build_sam3_video_model``) rather than the
    multi-GPU predictor wrapper to avoid unnecessary distributed overhead.
    """

    supports_text_prompts: bool = True

    def __init__(
        self,
        checkpoint: str,
        bpe_path: Optional[str],
        device: str,
        image_dir: Optional[str],
        num_objects: int,
        *,
        shared_model=None,
    ) -> None:
        if shared_model is not None:
            self._model = shared_model
        else:
            self._model = self._build_model(checkpoint, bpe_path, device)

        self._device = device
        self._image_dir = image_dir
        self._num_objects = num_objects

        # Internal tracking state
        self._state: Optional[dict] = None
        self._propagation_gen = None
        self._curr_ti: int = -1
        self._reverse: bool = False
        self._object_ids: set = set()

        # Anchor bookkeeping
        self._permanent_anchors: Dict[int, Tuple[torch.Tensor, List[int]]] = {}
        self._all_anchors: Dict[int, Tuple[torch.Tensor, List[int]]] = {}
        self._text_prompts: Dict[tuple, str] = {}  # (frame_idx, obj_id) → text

    @staticmethod
    def _build_model(checkpoint, bpe_path, device):
        from sam3.model_builder import build_sam3_video_model

        # Allow null/empty checkpoint to trigger HuggingFace auto-download
        ckpt_path = checkpoint if checkpoint else None
        load_hf = ckpt_path is None
        return build_sam3_video_model(
            checkpoint_path=ckpt_path,
            load_from_HF=load_hf,
            bpe_path=bpe_path,
            strict_state_dict_loading=False,
            device=device,
        )

    # -- State management ------------------------------------------------------

    def _ensure_state(self) -> None:
        if self._state is None:
            if self._image_dir is None:
                raise RuntimeError(
                    "SAM 3 backend requires image_dir (workspace/images) "
                    "to be set before the first step()."
                )
            self._state = self._model.init_state(
                resource_path=self._image_dir,
            )
            log.info("SAM 3 state initialised from %s", self._image_dir)

    def _invalidate_generator(self) -> None:
        self._propagation_gen = None

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
        self._ensure_state()
        if frame_idx is not None:
            self._curr_ti = frame_idx
        else:
            self._curr_ti += 1

        if mask is not None:
            return self._step_with_mask(
                mask, objects, idx_mask=idx_mask,
                force_permanent=force_permanent,
            )
        else:
            return self._step_propagate()

    def _step_with_mask(
        self,
        mask: torch.Tensor,
        objects: Optional[List[int]],
        *,
        idx_mask: bool,
        force_permanent: bool,
    ) -> torch.Tensor:
        """Register an anchor frame — convert mask to point prompts for SAM 3."""
        if objects is None:
            objects = list(range(1, self._num_objects + 1))

        binary_masks = self._to_binary_masks(mask, objects, idx_mask)

        for obj_id, bmask in binary_masks.items():
            points, labels = self._mask_to_points(bmask)
            if points is None:
                continue
            # SAM 3 add_prompt with points (normalized coords by default)
            H, W = bmask.shape[-2:]
            norm_points = [[p[0] / W, p[1] / H] for p in points]
            self._model.add_prompt(
                inference_state=self._state,
                frame_idx=self._curr_ti,
                points=norm_points,
                point_labels=labels,
                obj_id=obj_id,
            )
            self._object_ids.add(obj_id)

        # Store for permanent / re-anchor
        anchor_data = (mask.clone().cpu(), list(objects))
        self._all_anchors[self._curr_ti] = anchor_data
        if force_permanent:
            self._permanent_anchors[self._curr_ti] = anchor_data

        self._invalidate_generator()
        return self._binary_dict_to_prob(binary_masks)

    @staticmethod
    def _mask_to_points(
        binary_mask: torch.Tensor,
    ) -> Tuple[Optional[List[List[float]]], Optional[List[int]]]:
        """Convert a binary mask to point prompts (centre of mass + extremes).

        Returns (points, labels) where points are in *absolute* pixel
        coordinates ``[[x, y], ...]`` and labels are all ``1`` (positive).
        The caller is responsible for normalising to [0, 1] if needed.
        """
        coords = torch.nonzero(binary_mask.squeeze() > 0.5)  # (N, 2) as (row, col)
        if len(coords) == 0:
            return None, None

        # Centre of mass
        center = coords.float().mean(dim=0)
        cy, cx = center[0].item(), center[1].item()
        points = [[cx, cy]]
        labels = [1]

        # Add extremes for better coverage if mask is large enough
        if len(coords) > 20:
            rows, cols = coords[:, 0], coords[:, 1]
            for r, c in [
                (rows.min().item(), cols[rows.argmin()].item()),  # top
                (rows.max().item(), cols[rows.argmax()].item()),  # bottom
                (rows[cols.argmin()].item(), cols.min().item()),  # left
                (rows[cols.argmax()].item(), cols.max().item()),  # right
            ]:
                points.append([float(c), float(r)])
                labels.append(1)

        return points, labels

    def _step_propagate(self) -> torch.Tensor:
        """Consume the next frame from the SAM 3 propagation generator."""
        if self._propagation_gen is None:
            self._propagation_gen = self._model.propagate_in_video(
                self._state, reverse=self._reverse,
            )

        try:
            frame_idx, outputs = next(self._propagation_gen)
            while frame_idx != self._curr_ti:
                log.debug(
                    "Skipping SAM 3 generator frame %d (expecting %d)",
                    frame_idx, self._curr_ti,
                )
                frame_idx, outputs = next(self._propagation_gen)
        except StopIteration:
            log.warning(
                "SAM 3 propagation generator exhausted at frame %d",
                self._curr_ti,
            )
            prob = torch.zeros(
                self._num_objects + 1, 1, 1, device=self._device,
            )
            prob[0] = 1.0
            return prob

        return self._sam3_outputs_to_prob(outputs)

    # -- Text prompt support ---------------------------------------------------

    def add_text_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        text: str,
    ) -> torch.Tensor:
        """Segment all instances of *text* at *frame_idx*.

        Returns:
            (num_objects+1, H, W) probability tensor.
        """
        self._ensure_state()

        frame_idx_out, outputs = self._model.add_prompt(
            inference_state=self._state,
            frame_idx=frame_idx,
            text_str=text,
        )
        self._object_ids.add(obj_id)
        self._text_prompts[(frame_idx, obj_id)] = text
        self._invalidate_generator()

        return self._sam3_outputs_to_prob(outputs)

    # -- Memory management -----------------------------------------------------

    def clear_memory(self) -> None:
        if self._state is not None:
            self._model.reset_state(self._state)
        self._permanent_anchors.clear()
        self._all_anchors.clear()
        self._object_ids.clear()
        self._text_prompts.clear()
        self._invalidate_generator()
        self._curr_ti = -1

    def clear_non_permanent_memory(self) -> None:
        if self._state is not None:
            self._model.reset_state(self._state)
        self._all_anchors = dict(self._permanent_anchors)
        self._invalidate_generator()
        self._curr_ti = -1
        # Re-add permanent anchors
        for fi, (mask, objects) in sorted(self._permanent_anchors.items()):
            binary_masks = self._to_binary_masks(
                mask.to(self._device), objects, idx_mask=True,
            )
            for obj_id, bmask in binary_masks.items():
                points, labels = self._mask_to_points(bmask)
                if points is None:
                    continue
                H, W = bmask.shape[-2:]
                norm_points = [[p[0] / W, p[1] / H] for p in points]
                self._model.add_prompt(
                    inference_state=self._state,
                    frame_idx=fi,
                    points=norm_points,
                    point_labels=labels,
                    obj_id=obj_id,
                )

    def clear_sensory_memory(self) -> None:
        self._invalidate_generator()
        self._curr_ti = -1

    def update_config(self, cfg: Any) -> None:
        pass  # SAM 3 has no runtime-tunable memory parameters

    def delete_objects(self, objects: List[int]) -> None:
        for oid in objects:
            self._object_ids.discard(oid)
            if self._state is not None:
                try:
                    self._model.remove_object(
                        inference_state=self._state, obj_id=oid,
                    )
                except (AttributeError, KeyError):
                    pass

    def output_prob_to_mask(self, output_prob: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output_prob, dim=0)

    def get_memory_status(self) -> MemoryStatus:
        return MemoryStatus(
            perm_tokens=len(self._permanent_anchors),
            work_tokens=len(self._all_anchors) - len(self._permanent_anchors),
            max_work_tokens=max(6, len(self._all_anchors)),
            long_tokens=0,
            max_long_tokens=1,
        )

    # -- Propagation direction -------------------------------------------------

    def set_propagation_direction(self, reverse: bool) -> None:
        self._reverse = reverse

    # -- Conversion helpers ----------------------------------------------------

    def _sam3_outputs_to_prob(
        self, outputs: dict,
    ) -> torch.Tensor:
        """Convert SAM 3 output dict to (N+1, H, W) probability tensor."""
        mask_dict: Dict[int, torch.Tensor] = {}
        if outputs is not None and "out_binary_masks" in outputs:
            obj_ids = outputs["out_obj_ids"]   # numpy int64 array
            masks = outputs["out_binary_masks"]  # numpy bool (N, H, W)
            for i, oid in enumerate(obj_ids):
                mask_dict[int(oid)] = torch.from_numpy(
                    masks[i].astype(np.float32),
                )
        return self._binary_dict_to_prob(mask_dict)

    @staticmethod
    def _to_binary_masks(
        mask: torch.Tensor,
        objects: List[int],
        idx_mask: bool,
    ) -> Dict[int, torch.Tensor]:
        result: Dict[int, torch.Tensor] = {}
        if idx_mask:
            for obj_id in objects:
                result[obj_id] = (mask == obj_id).float()
        else:
            for i, obj_id in enumerate(objects):
                if i < mask.shape[0]:
                    result[obj_id] = (mask[i] > 0.5).float()
        return result

    def _binary_dict_to_prob(
        self, mask_dict: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        if not mask_dict:
            return torch.zeros(
                self._num_objects + 1, 1, 1, device=self._device,
            )

        sample = next(iter(mask_dict.values()))
        H, W = sample.shape[-2:]

        prob = torch.zeros(self._num_objects + 1, H, W, device=self._device)
        for obj_id, m in mask_dict.items():
            if 1 <= obj_id <= self._num_objects:
                prob[obj_id] = m.view(H, W).to(self._device)
        fg_max = (
            prob[1:].max(dim=0).values
            if prob.shape[0] > 1
            else torch.zeros(H, W, device=self._device)
        )
        prob[0] = 1.0 - fg_max
        return prob


# ── Click segmentation ───────────────────────────────────────────────────────


class Sam3ClickBackend:
    """Click-based single-frame segmentation using SAM 3's interactive
    image predictor (``SAM3InteractiveImagePredictor``).

    Shares the tracker backbone from the video model so that no extra
    model weights are loaded.
    """

    def __init__(
        self,
        tracker_model,
        device: str,
    ) -> None:
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

        self._predictor = SAM3InteractiveImagePredictor(tracker_model)
        self._predictor.to(device)
        self._device = device
        self._anchored = False
        self._points: List[List[int]] = []
        self._labels: List[int] = []

    def interact(
        self,
        image: torch.Tensor,
        x: int,
        y: int,
        is_positive: bool,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self._anchored:
            # Convert (1, 3, H, W) float [0,1] → (H, W, 3) uint8
            if image.dim() == 4:
                img = image.squeeze(0)
            else:
                img = image
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            self._predictor.set_image(img_np)
            self._points = []
            self._labels = []
            self._anchored = True

        self._points.append([x, y])
        self._labels.append(1 if is_positive else 0)

        masks, scores, logits = self._predictor.predict(
            point_coords=np.array(self._points),
            point_labels=np.array(self._labels),
            multimask_output=False,
        )
        # masks: (1, H, W) bool → (1, 1, H, W) float
        return (
            torch.from_numpy(masks[0:1])
            .float()
            .unsqueeze(0)
            .to(self._device)
        )

    def unanchor(self) -> None:
        self._anchored = False
        self._points = []
        self._labels = []

    def undo(self) -> Optional[torch.Tensor]:
        if not self._points:
            return None
        self._points.pop()
        self._labels.pop()
        if not self._points:
            return None
        masks, _, _ = self._predictor.predict(
            point_coords=np.array(self._points),
            point_labels=np.array(self._labels),
            multimask_output=False,
        )
        return (
            torch.from_numpy(masks[0:1])
            .float()
            .unsqueeze(0)
            .to(self._device)
        )
