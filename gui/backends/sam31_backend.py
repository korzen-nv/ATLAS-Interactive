"""SAM 3.1 (Object Multiplex) backend — uses the multiplex tracker for
batched multi-object mask propagation, and SAM3InteractiveImagePredictor
(via an adapter) for click segmentation.

SAM 3.1 groups tracked objects into fixed-capacity buckets and processes
them jointly, yielding significant speedups for multi-object scenarios.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gui.backends.base import MemoryStatus
from gui.backends.virtual_frames import VirtualFrameProxy

log = logging.getLogger(__name__)


# ── Adapter for SAM3InteractiveImagePredictor ──────────────────────────────


class _MultiplexTrackerAdapter:
    """Makes a ``VideoTrackingDynamicMultiplex`` model look like a
    ``Sam3TrackerBase`` for ``SAM3InteractiveImagePredictor``.

    Only the attributes/methods actually accessed by the predictor's
    ``set_image`` and ``predict`` paths are implemented.
    """

    def __init__(self, multiplex_model) -> None:
        self._m = multiplex_model

    # -- Attributes expected by SAM3InteractiveImagePredictor -----------------

    @property
    def image_size(self):
        return self._m.image_size

    @property
    def sam_prompt_encoder(self):
        return self._m.interactive_sam_prompt_encoder

    @property
    def sam_mask_decoder(self):
        return self._m.interactive_sam_mask_decoder

    @property
    def no_mem_embed(self):
        return self._m.interactivity_no_mem_embed

    @property
    def device(self):
        return next(self._m.parameters()).device

    num_feature_levels = 3

    # -- Methods expected by SAM3InteractiveImagePredictor --------------------

    def forward_image(self, img_batch):
        """Run the backbone and return interactive-branch features in the
        plain-tensor format that ``Sam3TrackerBase`` would return."""
        from sam3.model.data_misc import NestedTensor

        if not isinstance(img_batch, NestedTensor):
            img_batch = NestedTensor(img_batch, None)
        backbone_out = self._m.forward_image(
            img_batch, need_interactive_out=True,
        )
        # Extract the interactive branch and unwrap NestedTensors → tensors
        interactive = backbone_out["interactive"]
        plain = {
            "backbone_fpn": [nt.tensors for nt in interactive["backbone_fpn"]],
            "vision_pos_enc": list(interactive["vision_pos_enc"]),
        }
        return plain

    def _prepare_backbone_features(self, backbone_out):
        """Flatten visual features into ``(HW, N, C)`` format — mirrors
        ``Sam3TrackerBase._prepare_backbone_features``."""
        backbone_out = backbone_out.copy()
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    @staticmethod
    def _maybe_clone(t):
        return t


# ── Propagation ──────────────────────────────────────────────────────────────


class Sam31PropagationBackend:
    """Wraps the SAM 3.1 multiplex tracker for batched multi-object
    mask propagation using ``add_new_masks`` / ``propagate_in_video``.
    """

    supports_text_prompts: bool = False

    def __init__(
        self,
        checkpoint: str,
        device: str,
        image_dir: Optional[str],
        num_objects: int,
        *,
        shared_model=None,
    ) -> None:
        if shared_model is not None:
            self._model = shared_model
        else:
            self._model = self._build_model(checkpoint, device)
        self._patch_get_image_feature(self._model)

        self._device = device
        self._image_dir = image_dir
        self._num_objects = num_objects

        # Tracker inference state
        self._state: Optional[dict] = None
        self._propagation_gen = None
        self._curr_ti: int = -1
        self._reverse: bool = False
        self._object_ids: set = set()

        # Anchor bookkeeping
        self._permanent_anchors: Dict[int, Tuple[torch.Tensor, List[int]]] = {}
        self._all_anchors: Dict[int, Tuple[torch.Tensor, List[int]]] = {}

    @staticmethod
    def _build_model(checkpoint, device):
        from sam3.model_builder import build_sam3_multiplex_video_model

        ckpt_path = checkpoint if checkpoint else None
        load_hf = ckpt_path is None

        # Build model without loading checkpoint — we handle loading ourselves
        # because HF/predictor checkpoints use "tracker.model.*" prefixed keys
        # for the tracker portion, while the standalone model expects bare keys.
        model = build_sam3_multiplex_video_model(
            checkpoint_path=None,
            load_from_HF=False,
            strict_state_dict_loading=False,
            device=device,
        )

        if load_hf:
            from sam3.model_builder import download_ckpt_from_hf
            ckpt_path = download_ckpt_from_hf(version="sam3.1")

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]

            # Detect incompatible SAM 3 checkpoints: they use "tracker.*"
            # keys (without ".model." infix) and a different architecture.
            has_tracker_model = any(
                k.startswith("tracker.model.") for k in ckpt)
            has_tracker_bare = any(
                k.startswith("tracker.") and not k.startswith("tracker.model.")
                for k in ckpt)
            if has_tracker_bare and not has_tracker_model:
                raise ValueError(
                    f"Checkpoint '{ckpt_path}' appears to be a SAM 3 "
                    f"checkpoint (tracker.* keys without tracker.model.* "
                    f"prefix).  SAM 3 checkpoints are not compatible with "
                    f"the SAM 3.1 multiplex architecture.  Use 'backend: "
                    f"sam3' for this checkpoint, or use a SAM 3.1 checkpoint "
                    f"(e.g. sam3.1_multiplex.pt) with 'backend: sam31'."
                )

            # Remap predictor-style checkpoint keys.  The HF checkpoint is
            # structured for the full Sam3MultiplexTrackingWithInteractivity
            # model (tracker.model.* + detector.*).  We load the tracker
            # weights directly and pull the vision backbone from the detector
            # (since the predictor builder deletes the tracker's backbone).
            tracker_prefix = "tracker.model."
            det_vis_prefix = "detector.backbone.vision_backbone."
            if has_tracker_model:
                remapped = {}
                for k, v in ckpt.items():
                    if k.startswith(tracker_prefix):
                        remapped[k[len(tracker_prefix):]] = v
                    elif k.startswith(det_vis_prefix):
                        remapped["backbone.vision_backbone."
                                 + k[len(det_vis_prefix):]] = v
                ckpt = remapped

            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            if missing:
                log.warning("SAM 3.1 missing keys (%d): %s…",
                            len(missing), missing[:5])
            if unexpected:
                log.warning("SAM 3.1 unexpected keys (%d): %s…",
                            len(unexpected), unexpected[:5])

        model.to(device=device)
        return model

    # -- State management ------------------------------------------------------

    def _ensure_state(self) -> None:
        """Lazily initialise the tracker state from the image directory."""
        if self._state is None:
            if self._image_dir is None:
                raise RuntimeError(
                    "SAM 3.1 backend requires image_dir (workspace/images) "
                    "to be set before the first step()."
                )
            # Sam3VideoTrackingMultiplexDemo.init_state expects pre-loaded
            # frame metadata — load frames ourselves via SAM 3.1's io_utils.
            from sam3.model.io_utils import load_video_frames

            images, video_height, video_width = load_video_frames(
                video_path=self._image_dir,
                image_size=self._model.image_size,
                offload_video_to_cpu=True,
                async_loading_frames=True,
            )
            self._state = self._model.init_state(
                video_height=video_height,
                video_width=video_width,
                num_frames=len(images),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            self._state["images"] = images
            log.info("SAM 3.1 multiplex state initialised from %s", self._image_dir)

    @staticmethod
    def _patch_get_image_feature(model) -> None:
        """Patch _get_image_feature to skip need_sam3_out.

        The default demo method passes need_sam3_out=True which causes
        TriHeadVisionOnly to put flat tensor entries into backbone_out
        alongside the nested dicts.  The clone loop in forward_image then
        fails because it treats every key as a nested dict.  We only need
        the interactive + propagation heads for tracking.
        """
        from sam3.model.data_misc import NestedTensor

        original = type(model)._get_image_feature

        def _patched(self, inference_state, frame_idx, batch_size):
            image, backbone_out = inference_state["cached_features"].get(
                frame_idx, (None, None),
            )
            if backbone_out is None:
                image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)
                backbone_out = self.forward_image(
                    NestedTensor(tensors=image, mask=None),
                    need_sam3_out=False,
                    need_interactive_out=True,
                    need_propagation_out=True,
                )
                inference_state["cached_features"] = {
                    frame_idx: (image, backbone_out),
                }
            features = self._prepare_backbone_features(backbone_out)
            return image, features

        import types
        model._get_image_feature = types.MethodType(_patched, model)

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
        if not binary_masks:
            return self._binary_dict_to_prob({})

        # Store as index mask for consistent replay
        if idx_mask:
            idx_mask_store = mask.clone().cpu()
        else:
            # Convert probability mask (num_objects, H, W) → index mask (H, W)
            bg = 1.0 - mask.max(dim=0).values
            idx_mask_store = torch.argmax(
                torch.cat([bg.unsqueeze(0), mask], dim=0), dim=0,
            ).cpu()
        anchor_data = (idx_mask_store, list(objects))
        self._all_anchors[self._curr_ti] = anchor_data
        if force_permanent:
            self._permanent_anchors[self._curr_ti] = anchor_data

        # The multiplex model's add_new_masks fails when multiplex_state
        # already exists but the target frame has no output yet (it
        # incorrectly sets add_to_existing_state=True).  Reset tracking
        # and replay all anchors so every frame is an init conditioning
        # frame with a fresh multiplex_state.
        self._model._reset_tracking_results(self._state)
        self._object_ids.clear()
        self._replay_anchors()

        self._invalidate_generator()
        return self._binary_dict_to_prob(binary_masks)

    def _replay_anchors(self) -> None:
        """Re-add all stored anchors into a clean tracking state.

        Each anchor is added with a fresh ``multiplex_state`` so that
        ``add_new_masks`` always takes the init-conditioning-frame path
        (which doesn't require pre-existing output on the frame).

        All anchors use the *union* of object IDs across every frame
        (with zero masks for absent objects) so that every
        ``multiplex_state`` created has the same ``total_valid_entries``.
        """
        if not self._all_anchors:
            return

        # Collect the union of all object IDs and a reference spatial size
        all_obj_ids: set = set()
        spatial_shape = None
        for mask, objects in self._all_anchors.values():
            for obj_id in objects:
                bmask = (mask == obj_id)
                if bmask.any():
                    all_obj_ids.add(obj_id)
                    if spatial_shape is None:
                        spatial_shape = mask.shape[-2:]
        if not all_obj_ids or spatial_shape is None:
            return
        all_obj_ids_sorted = sorted(all_obj_ids)

        for fi, (mask, objects) in sorted(self._all_anchors.items()):
            binary_masks = self._to_binary_masks(
                mask.to(self._device), objects, idx_mask=True,
            )
            # Build a tensor for ALL objects — zero for absent ones
            masks_list = []
            for oid in all_obj_ids_sorted:
                if oid in binary_masks:
                    masks_list.append(binary_masks[oid])
                else:
                    masks_list.append(torch.zeros(
                        spatial_shape, device=self._device,
                    ))
            # Reset multiplex_state so add_new_masks treats this as a
            # fresh init frame (is_new_state=True → add_to_existing_state=False).
            self._state["multiplex_state"] = None
            masks_tensor = torch.stack(masks_list, dim=0)
            self._model.add_new_masks(
                self._state, fi, all_obj_ids_sorted, masks_tensor,
            )
        self._object_ids = set(all_obj_ids_sorted)

    def _step_propagate(self) -> torch.Tensor:
        """Consume the next frame from the multiplex propagation generator."""
        if self._propagation_gen is None:
            self._model.propagate_in_video_preflight(self._state)
            self._propagation_gen = self._model.propagate_in_video(
                self._state,
                start_frame_idx=self._curr_ti,
                max_frame_num_to_track=None,
                reverse=self._reverse,
            )

        try:
            result = next(self._propagation_gen)
            # Multiplex yields 4-tuple: (frame_idx, obj_ids, low_res_masks,
            #                             video_res_masks)
            frame_idx = result[0]
            obj_ids = result[1]
            video_res_masks = result[3]

            while frame_idx != self._curr_ti:
                log.debug(
                    "Skipping SAM 3.1 tracker frame %d (expecting %d)",
                    frame_idx, self._curr_ti,
                )
                result = next(self._propagation_gen)
                frame_idx = result[0]
                obj_ids = result[1]
                video_res_masks = result[3]

        except StopIteration:
            log.warning(
                "SAM 3.1 propagation generator exhausted at frame %d",
                self._curr_ti,
            )
            prob = torch.zeros(
                self._num_objects + 1, 1, 1, device=self._device,
            )
            prob[0] = 1.0
            return prob

        # video_res_masks are logit scores — binarize at 0
        mask_dict: Dict[int, torch.Tensor] = {}
        for i, oid in enumerate(obj_ids):
            mask_dict[int(oid)] = (video_res_masks[i] > 0).squeeze(0).float()
        return self._binary_dict_to_prob(mask_dict)

    # -- Global memory injection -------------------------------------------------

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
        """Resize and normalize a (3, H, W) [0,1] tensor for SAM 3.1 state."""
        img_size = self._model.image_size
        img = F.interpolate(
            image.unsqueeze(0), size=(img_size, img_size),
            mode='bilinear', align_corners=False,
        ).squeeze(0)
        img = (img - 0.5) / 0.5
        return img.half().cpu()

    # -- Memory management -----------------------------------------------------

    def clear_memory(self) -> None:
        if self._state is not None:
            if isinstance(self._state["images"], VirtualFrameProxy):
                self._state["images"].clear_virtual()
            self._model.clear_all_points_in_video(self._state)
        self._permanent_anchors.clear()
        self._all_anchors.clear()
        self._object_ids.clear()
        self._invalidate_generator()
        self._curr_ti = -1

    def clear_non_permanent_memory(self) -> None:
        if self._state is not None:
            self._model._reset_tracking_results(self._state)
            self._state["multiplex_state"] = None
        self._all_anchors = dict(self._permanent_anchors)
        self._object_ids.clear()
        self._invalidate_generator()
        self._curr_ti = -1
        if self._state is not None:
            self._replay_anchors()

    def clear_sensory_memory(self) -> None:
        self._invalidate_generator()
        self._curr_ti = -1

    def update_config(self, cfg: Any) -> None:
        pass  # SAM 3.1 has no runtime-tunable memory parameters

    def delete_objects(self, objects: List[int]) -> None:
        if self._state is None or not objects:
            return
        for oid in objects:
            self._object_ids.discard(oid)
            self._model.remove_object(self._state, oid, strict=False)
        self._invalidate_generator()

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

    @staticmethod
    def _to_binary_masks(
        mask: torch.Tensor,
        objects: List[int],
        idx_mask: bool,
    ) -> Dict[int, torch.Tensor]:
        result: Dict[int, torch.Tensor] = {}
        if idx_mask:
            for obj_id in objects:
                bmask = (mask == obj_id).float()
                if bmask.any():
                    result[obj_id] = bmask
        else:
            for i, obj_id in enumerate(objects):
                if i < mask.shape[0]:
                    bmask = (mask[i] > 0.5).float()
                    if bmask.any():
                        result[obj_id] = bmask
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


class Sam31ClickBackend:
    """Click-based single-frame segmentation using SAM 3's interactive
    image predictor via the multiplex adapter.

    Shares the multiplex model's backbone so no extra weights are loaded.
    """

    def __init__(
        self,
        multiplex_model,
        device: str,
    ) -> None:
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

        adapter = _MultiplexTrackerAdapter(multiplex_model)
        self._predictor = SAM3InteractiveImagePredictor(adapter)
        self._predictor.to(device)
        self._device = device
        self._anchored = False
        self._points: List[List[int]] = []
        self._labels: List[int] = []
        self._logits: Optional[np.ndarray] = None

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
