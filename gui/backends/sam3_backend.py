"""SAM 3 backend — extends Sam2PropagationBackend with text prompt support."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from gui.backends.sam2_backend import Sam2PropagationBackend

log = logging.getLogger(__name__)


class Sam3PropagationBackend(Sam2PropagationBackend):
    """SAM 3 video predictor with text prompt capability.

    Backward-compatible with all SAM 2 point/box interactions.
    Adds ``add_text_prompt()`` for concept-level segmentation.
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
        shared_predictor=None,
    ) -> None:
        # Skip Sam2PropagationBackend.__init__ — we have a different builder
        self._bpe_path = bpe_path

        if shared_predictor is not None:
            self._predictor = shared_predictor
        else:
            self._predictor = self._build_predictor(checkpoint, bpe_path, device)

        self._device = device
        self._image_dir = image_dir
        self._num_objects = num_objects

        self._state: Optional[dict] = None
        self._propagation_gen = None
        self._curr_ti: int = -1
        self._reverse: bool = False
        self._object_ids: set = set()

        self._permanent_anchors: dict = {}
        self._all_anchors: dict = {}
        self._text_prompts: Dict[tuple, str] = {}  # (frame_idx, obj_id) → text

    @staticmethod
    def _build_predictor(checkpoint, bpe_path, device):
        from sam3.build_sam import build_sam3_video_predictor
        return build_sam3_video_predictor(
            checkpoint_path=checkpoint,
            bpe_path=bpe_path,
            device=device,
        )

    # -- Text prompt support ---------------------------------------------------

    def add_text_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        text: str,
    ) -> torch.Tensor:
        """Segment all instances of a text-described concept at a given frame.

        Args:
            frame_idx: Frame to apply the prompt to.
            obj_id: Object ID to assign to the detected instances.
            text: Natural language description (e.g. "grasper").

        Returns:
            (num_objects+1, H, W) probability tensor.
        """
        self._ensure_state()

        frame_idx_out, outputs = self._predictor.add_prompt(
            self._state,
            frame_idx=frame_idx,
            text_str=text,
            obj_id=obj_id,
        )
        self._object_ids.add(obj_id)
        self._text_prompts[(frame_idx, obj_id)] = text
        self._invalidate_generator()

        # Convert SAM 3 output to standard probability format
        if "out_binary_masks" in outputs:
            binary_mask = outputs["out_binary_masks"][0].float()  # (H, W)
            mask_dict = {obj_id: binary_mask.unsqueeze(0)}
        elif "out_mask_logits" in outputs:
            mask_logits = outputs["out_mask_logits"][0]  # (1, H, W)
            binary_mask = (mask_logits > 0).float()
            mask_dict = {obj_id: binary_mask}
        else:
            log.warning("SAM 3 add_prompt returned no masks")
            mask_dict = {}

        return self._binary_dict_to_prob(mask_dict)
