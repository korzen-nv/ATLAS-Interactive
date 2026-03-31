"""SAM 2 backend — adapts SAM2VideoPredictor / SAM2ImagePredictor to the
PropagationBackend and ClickBackend protocols.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gui.backends.base import MemoryStatus
from gui.backends.virtual_frames import VirtualFrameProxy

log = logging.getLogger(__name__)


# ── Multi-threaded frame loader ──────────────────────────────────────────────


class _MultiThreadFrameLoader:
    """Drop-in replacement for SAM 2's ``AsyncVideoFrameLoader`` that loads
    JPEG frames in parallel using a :class:`~concurrent.futures.ThreadPoolExecutor`.

    The original loader uses a single background thread, which means JPEG
    decompression + resize is serialised.  PIL releases the GIL during
    image I/O so multiple threads provide a real speed-up.

    Interface is identical to ``AsyncVideoFrameLoader`` (``__getitem__`` /
    ``__len__`` plus ``.video_height`` / ``.video_width``), so SAM 2's
    ``init_state`` can use it transparently.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
        *,
        num_workers: Optional[int] = None,
    ):
        from sam2.utils.misc import _load_img_as_tensor

        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.compute_device = compute_device
        self.images: list = [None] * len(img_paths)
        self.exception: Optional[Exception] = None
        self.video_height: Optional[int] = None
        self.video_width: Optional[int] = None
        self._load_img = _load_img_as_tensor

        if num_workers is None:
            num_workers = min(os.cpu_count() or 4, len(img_paths), 8)

        # Frame 0 must be ready before init_state returns (SAM 2 warms up
        # the backbone on it).
        self._load_frame(0)

        # Submit the rest to the thread pool.
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="sam2-frame-loader",
        )
        self._futures: dict = {}
        for n in range(1, len(img_paths)):
            self._futures[n] = self._executor.submit(self._load_frame, n)

        log.info(
            "Loading %d frames with %d worker threads", len(img_paths), num_workers,
        )

    # -- internal ----------------------------------------------------------

    def _load_frame(self, index: int) -> torch.Tensor:
        img, h, w = self._load_img(self.img_paths[index], self.image_size)
        if self.video_height is None:
            self.video_height = h
            self.video_width = w
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img
        return img

    # -- public interface (same as AsyncVideoFrameLoader) ------------------

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        # Block until this specific frame is ready.
        future = self._futures.get(index)
        if future is not None:
            try:
                return future.result()
            except Exception as e:
                self.exception = e
                raise RuntimeError("Failure in frame loading thread") from e

        # Fallback: load directly (should not happen).
        return self._load_frame(index)

    def __len__(self) -> int:
        return len(self.images)


def _init_hydra_for_sam2() -> None:
    """(Re-)initialise Hydra with SAM 2's config directory.

    The app initialises Hydra at startup for CUTIE's config.  SAM 2's
    ``build_sam2`` calls ``compose()`` which needs Hydra pointed at SAM's
    own config dir, so we clear and reinitialise.
    """
    import os
    import sam2
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    sam2_cfg_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
    # initialize_config_dir is a context manager but we intentionally leave
    # Hydra initialised so the subsequent compose() call works.
    initialize_config_dir(config_dir=sam2_cfg_dir, version_base="1.2")


# ── Propagation ──────────────────────────────────────────────────────────────


class Sam2PropagationBackend:
    """Wraps SAM2VideoPredictor to satisfy PropagationBackend.

    Lifecycle:
        1. Constructed with the path to a JPEG frame directory.
        2. ``init_state`` is called lazily on the first ``step()``.
        3. Anchor frames (with mask) are registered via ``add_new_mask``.
        4. Propagation frames (without mask) consume a generator from
           ``propagate_in_video``.
    """

    def __init__(
        self,
        checkpoint: str,
        model_cfg: Optional[str],
        device: str,
        image_dir: Optional[str],
        num_objects: int,
        *,
        shared_predictor=None,
    ) -> None:
        if shared_predictor is not None:
            self._predictor = shared_predictor
        else:
            self._predictor = self._build_predictor(checkpoint, model_cfg, device)

        self._device = device
        self._image_dir = image_dir
        self._num_objects = num_objects

        # Internal tracking state
        self._state: Optional[dict] = None
        self._propagation_gen = None
        self._curr_ti: int = -1
        self._reverse: bool = False
        self._object_ids: set = set()

        # Permanent memory approximation
        self._permanent_anchors: Dict[int, Tuple[torch.Tensor, List[int]]] = {}
        self._all_anchors: Dict[int, Tuple[torch.Tensor, List[int]]] = {}

    @staticmethod
    def _build_predictor(checkpoint, model_cfg, device):
        from sam2.build_sam import build_sam2_video_predictor
        _init_hydra_for_sam2()
        return build_sam2_video_predictor(
            model_cfg, checkpoint, device=device,
        )

    # -- State management ------------------------------------------------------

    def _ensure_state(self) -> None:
        if self._state is None:
            if self._image_dir is None:
                raise RuntimeError(
                    "SAM 2 backend requires image_dir (workspace/images) "
                    "to be set before the first step()."
                )
            # Temporarily replace SAM 2's single-thread loader with our
            # multi-threaded version so frames are decoded in parallel.
            import sam2.utils.misc as _sam2_misc

            _orig_loader = _sam2_misc.AsyncVideoFrameLoader
            _sam2_misc.AsyncVideoFrameLoader = _MultiThreadFrameLoader
            try:
                self._state = self._predictor.init_state(
                    video_path=self._image_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                    async_loading_frames=True,
                )
            finally:
                _sam2_misc.AsyncVideoFrameLoader = _orig_loader
            log.info(f"SAM 2 state initialised from {self._image_dir}")

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
        """Register an anchor frame with a user-provided mask."""
        if objects is None:
            objects = list(range(1, self._num_objects + 1))

        binary_masks = self._to_binary_masks(mask, objects, idx_mask)

        for obj_id, bmask in binary_masks.items():
            self._predictor.add_new_mask(
                self._state, self._curr_ti, obj_id, bmask,
            )
            self._object_ids.add(obj_id)

        # Store for permanent / re-anchor
        anchor_data = (mask.clone().cpu(), list(objects))
        self._all_anchors[self._curr_ti] = anchor_data
        if force_permanent:
            self._permanent_anchors[self._curr_ti] = anchor_data

        self._invalidate_generator()
        return self._binary_dict_to_prob(binary_masks)

    def _step_propagate(self) -> torch.Tensor:
        """Consume the next frame from the SAM propagation generator.

        Advances the generator until the yielded frame index matches
        ``self._curr_ti``, ensuring the returned mask corresponds to the
        frame the controller is currently displaying.
        """
        if self._propagation_gen is None:
            self._propagation_gen = self._predictor.propagate_in_video(
                self._state,
                start_frame_idx=self._curr_ti,
                reverse=self._reverse,
            )

        try:
            frame_idx, obj_ids, masks = next(self._propagation_gen)
            # Skip frames until we reach the one the controller expects.
            while frame_idx != self._curr_ti:
                log.debug(
                    "Skipping SAM 2 generator frame %d (expecting %d)",
                    frame_idx, self._curr_ti,
                )
                frame_idx, obj_ids, masks = next(self._propagation_gen)
        except StopIteration:
            log.warning("SAM 2 propagation generator exhausted at frame %d",
                        self._curr_ti)
            prob = torch.zeros(
                self._num_objects + 1, 1, 1, device=self._device,
            )
            prob[0] = 1.0
            return prob

        # masks: (num_objects, 1, H, W) bool tensor from SAM
        mask_dict = {}
        for i, oid in enumerate(obj_ids):
            mask_dict[int(oid)] = masks[i].squeeze(0).float()
        return self._binary_dict_to_prob(mask_dict)

    def inject_permanent_memory(self, image, mask, objects):
        self._ensure_state()
        if not isinstance(self._state["images"], VirtualFrameProxy):
            self._state["images"] = VirtualFrameProxy(self._state["images"])
        processed = self._prepare_foreign_image(image)
        virtual_idx = self._state["images"].add_virtual(processed)
        saved_ti = self._curr_ti
        self._curr_ti = virtual_idx
        self._step_with_mask(mask, objects, idx_mask=True, force_permanent=True)
        self._curr_ti = saved_ti

    def _prepare_foreign_image(self, image: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a (3, H, W) [0,1] tensor for SAM 2 state."""
        img_size = self._predictor.image_size
        img = F.interpolate(
            image.unsqueeze(0), size=(img_size, img_size),
            mode='bilinear', align_corners=False,
        ).squeeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        img = (img - mean) / std
        return img.cpu()

    def clear_memory(self) -> None:
        if self._state is not None:
            if isinstance(self._state["images"], VirtualFrameProxy):
                self._state["images"].clear_virtual()
            self._predictor.reset_state(self._state)
        self._permanent_anchors.clear()
        self._all_anchors.clear()
        self._object_ids.clear()
        self._invalidate_generator()
        self._curr_ti = -1

    def clear_non_permanent_memory(self) -> None:
        if self._state is not None:
            self._predictor.reset_state(self._state)
        self._all_anchors = dict(self._permanent_anchors)
        self._invalidate_generator()
        self._curr_ti = -1
        # Re-add permanent anchors
        for frame_idx, (mask, objects) in sorted(self._permanent_anchors.items()):
            binary_masks = self._to_binary_masks(mask.to(self._device), objects, idx_mask=True)
            for obj_id, bmask in binary_masks.items():
                self._predictor.add_new_mask(self._state, frame_idx, obj_id, bmask)

    def clear_sensory_memory(self) -> None:
        # SAM has no sensory memory — just reset the propagation generator
        self._invalidate_generator()
        self._curr_ti = -1

    def update_config(self, cfg: Any) -> None:
        pass  # SAM has no runtime-tunable memory parameters

    def delete_objects(self, objects: List[int]) -> None:
        for oid in objects:
            self._object_ids.discard(oid)
            if self._state is not None:
                try:
                    self._predictor.remove_object(self._state, oid)
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
        """Called by controller before propagation to set direction."""
        self._reverse = reverse

    # -- Mask conversion helpers -----------------------------------------------

    def _to_binary_masks(
        self,
        mask: torch.Tensor,
        objects: List[int],
        idx_mask: bool,
    ) -> Dict[int, torch.Tensor]:
        """Convert ATLAS mask to per-object (H, W) binary masks for SAM."""
        result = {}
        if idx_mask:
            for obj_id in objects:
                result[obj_id] = (mask == obj_id).float()
        else:
            # mask is (num_objects, H, W) probabilities
            for i, obj_id in enumerate(objects):
                if i < mask.shape[0]:
                    result[obj_id] = (mask[i] > 0.5).float()
        return result

    def _binary_dict_to_prob(
        self, mask_dict: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Convert {obj_id: (H, W) binary} → (N+1, H, W) probabilities."""
        if not mask_dict:
            return torch.zeros(self._num_objects + 1, 1, 1, device=self._device)

        sample = next(iter(mask_dict.values()))
        H, W = sample.shape[-2:]

        prob = torch.zeros(self._num_objects + 1, H, W, device=self._device)
        for obj_id, m in mask_dict.items():
            if 1 <= obj_id <= self._num_objects:
                prob[obj_id] = m.view(H, W)
        # Background = 1 - max(foreground)
        fg_max = prob[1:].max(dim=0).values if prob.shape[0] > 1 else torch.zeros(H, W, device=self._device)
        prob[0] = 1.0 - fg_max
        return prob


# ── Click segmentation ───────────────────────────────────────────────────────


class Sam2ClickBackend:
    """Wraps SAM2ImagePredictor to satisfy the ClickBackend protocol."""

    def __init__(
        self,
        checkpoint: str,
        model_cfg: Optional[str],
        device: str,
        *,
        shared_model=None,
    ) -> None:
        if shared_model is not None:
            self._predictor = self._wrap_image_predictor(shared_model)
        else:
            self._predictor = self._build_image_predictor(checkpoint, model_cfg, device)

        self._device = device
        self._anchored = False
        self._points: List[List[int]] = []
        self._labels: List[int] = []
        self._logits: Optional[np.ndarray] = None

    @staticmethod
    def _build_image_predictor(checkpoint, model_cfg, device):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _init_hydra_for_sam2()
        model = build_sam2(model_cfg, checkpoint, device=device)
        return SAM2ImagePredictor(model)

    @staticmethod
    def _wrap_image_predictor(model):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(model)

    def _prev_mask_to_logits(self, prev_mask: torch.Tensor) -> np.ndarray:
        """Convert (1, 1, H, W) probability mask to low-res logits for SAM."""
        mask_size = self._predictor.model.sam_prompt_encoder.mask_input_size
        pm = prev_mask.squeeze(0).float()  # (1, H, W)
        pm = F.interpolate(pm.unsqueeze(0), size=mask_size,
                           mode='bilinear', align_corners=False)
        pm = pm.squeeze(0).cpu().numpy()   # (1, mask_H, mask_W)
        pm = np.clip(pm, 1e-6, 1 - 1e-6)
        return np.log(pm / (1 - pm))       # logit transform

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
            self._logits = None
            self._anchored = True

        self._points.append([x, y])
        self._labels.append(1 if is_positive else 0)

        # Build mask_input: use stored logits from previous click,
        # or convert prev_mask on first click to seed refinement
        mask_input = None
        if self._logits is not None:
            mask_input = self._logits
        elif prev_mask is not None:
            mask_input = self._prev_mask_to_logits(prev_mask)

        masks, scores, logits = self._predictor.predict(
            point_coords=np.array(self._points),
            point_labels=np.array(self._labels),
            mask_input=mask_input,
            multimask_output=False,
        )
        self._logits = logits

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
        self._logits = None

    def undo(self) -> Optional[torch.Tensor]:
        if not self._points:
            return None
        self._points.pop()
        self._labels.pop()
        if not self._points:
            self._logits = None
            return None
        masks, _, logits = self._predictor.predict(
            point_coords=np.array(self._points),
            point_labels=np.array(self._labels),
            mask_input=self._logits,
            multimask_output=False,
        )
        self._logits = logits
        return (
            torch.from_numpy(masks[0:1])
            .float()
            .unsqueeze(0)
            .to(self._device)
        )
