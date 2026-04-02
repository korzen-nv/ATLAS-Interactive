from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional, Iterable, Dict, TYPE_CHECKING
import logging
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F

from gui.cutie.inference.memory_manager import MemoryManager
from gui.cutie.inference.object_manager import ObjectManager
from gui.cutie.inference.image_feature_store import ImageFeatureStore
from gui.cutie.model.cutie import CUTIE
from gui.cutie.utils.tensor_utils import pad_divide_by, unpad, aggregate

if TYPE_CHECKING:
    from gui.torch_rt import TorchRT

log = logging.getLogger()


def _infer_nonempty_prob_mask_objects(
        mask: torch.Tensor,
        preserve_objects: Optional[list[int]] = None) -> tuple[list[int], torch.Tensor]:
    flat_mask = mask.flatten(start_dim=1)
    active = flat_mask.amax(dim=1) > 0
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    preserved = []
    if preserve_objects is not None:
        preserved = sorted({obj for obj in preserve_objects if 1 <= obj <= mask.shape[0]})

    if active_indices.numel() == 0:
        if preserved:
            keep = torch.tensor([obj - 1 for obj in preserved], device=mask.device, dtype=torch.long)
            return preserved, mask.index_select(0, keep)
        objects = list(range(1, mask.shape[0] + 1))
        return objects, mask

    objects = (active_indices + 1).tolist()
    if preserved:
        objects = sorted(set(objects) | set(preserved))
        if objects == list(range(1, mask.shape[0] + 1)):
            return objects, mask
        keep = torch.tensor([obj - 1 for obj in objects], device=mask.device, dtype=torch.long)
        return objects, mask.index_select(0, keep)
    if active_indices.numel() == mask.shape[0]:
        return objects, mask
    return objects, mask.index_select(0, active_indices)


def _masks_by_tmp_id(
        mask: torch.Tensor,
        objects: list[int],
        corresponding_tmp_ids: list[int],
        *,
        idx_mask: bool) -> dict[int, torch.Tensor]:
    masks_by_tmp_id = {}
    for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
        if idx_mask:
            masks_by_tmp_id[tmp_id] = (mask == objects[mask_id])
        else:
            masks_by_tmp_id[tmp_id] = mask[mask_id]
    return masks_by_tmp_id


class _StepProfiler:
    """Lightweight CUDA-event profiler for InferenceCore.step().

    When disabled, all methods are no-ops (zero overhead).
    Enable with ``inference_core.profiler.enabled = True``.
    Results are printed every ``print_every`` frames.
    """
    def __init__(self, print_every: int = 50):
        self.enabled = False
        self.print_every = print_every
        self._events: list = []
        self._labels: list = []
        self._sections: list = []
        self._count = 0
        self._accum: Dict[str, float] = {}

    def mark(self, label: str) -> None:
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events.append(ev)
        self._labels.append(label)

    @contextmanager
    def section(self, label: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._sections.append((label, start, end))

    def finish_frame(self) -> None:
        if not self.enabled or len(self._events) < 2:
            self._events.clear()
            self._labels.clear()
            self._sections.clear()
            return
        torch.cuda.synchronize()
        for i in range(len(self._events) - 1):
            name = f"{self._labels[i]} → {self._labels[i+1]}"
            ms = self._events[i].elapsed_time(self._events[i + 1])
            self._accum[name] = self._accum.get(name, 0.0) + ms
        for name, start, end in self._sections:
            ms = start.elapsed_time(end)
            self._accum[name] = self._accum.get(name, 0.0) + ms
        total_name = f"TOTAL ({self._labels[0]} → {self._labels[-1]})"
        total_ms = self._events[0].elapsed_time(self._events[-1])
        self._accum[total_name] = self._accum.get(total_name, 0.0) + total_ms
        self._count += 1
        if self._count >= self.print_every:
            self._print()
        self._events.clear()
        self._labels.clear()
        self._sections.clear()

    def _print(self) -> None:
        n = self._count
        print(f"\n=== Step profiler ({n} frames) ===")
        for name, total_ms in self._accum.items():
            print(f"  {name:<40s} {total_ms / n:6.2f} ms/frame")
        print()
        self._accum.clear()
        self._count = 0


class InferenceCore:

    def __init__(self,
                 network: CUTIE,
                 cfg: DictConfig,
                 *,
                 image_feature_store: ImageFeatureStore = None,
                 torch_rt: Optional[TorchRT] = None,
                 trt_encoder=None,
                 trt_mask_decoder=None,
                 trt_readout=None):
        self.network = network
        self.cfg = cfg
        self.mem_every = cfg.mem_every
        stagger_updates = cfg.stagger_updates
        self.chunk_size = cfg.chunk_size
        self.save_aux = cfg.save_aux
        self.max_internal_size = cfg.max_internal_size
        self.flip_aug = cfg.flip_aug
        self.torch_rt = torch_rt

        self.curr_ti = -1
        self.last_mem_ti = 0
        # at which time indices should we update the sensory memory
        if stagger_updates >= self.mem_every:
            self.stagger_ti = set(range(1, self.mem_every + 1))
        else:
            self.stagger_ti = set(
                np.round(np.linspace(1, self.mem_every, stagger_updates)).astype(int))
        self.object_manager = ObjectManager()
        self.memory = MemoryManager(cfg=cfg, object_manager=self.object_manager)

        if image_feature_store is None:
            self.image_feature_store = ImageFeatureStore(
                self.network, trt_encoder=trt_encoder)
        else:
            self.image_feature_store = image_feature_store

        self.trt_mask_decoder = trt_mask_decoder
        self.trt_readout = trt_readout
        self.profiler = _StepProfiler()
        self.profiler.enabled = cfg.get('profile', False)
        self.last_mask = None

    def _resolve_trt_mask_decoder(self, num_objects: int):
        trt_mask_decoder = self.trt_mask_decoder
        if trt_mask_decoder is None:
            return None
        if hasattr(trt_mask_decoder, 'get'):
            return trt_mask_decoder.get(num_objects)
        return trt_mask_decoder

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(cfg=self.cfg, object_manager=self.object_manager)

    def clear_non_permanent_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_sensory_memory()

    def update_config(self, cfg):
        self.mem_every = cfg['mem_every']
        self.memory.update_config(cfg)

    def _add_memory(self,
                    image: torch.Tensor,
                    pix_feat: torch.Tensor,
                    prob: torch.Tensor,
                    key: torch.Tensor,
                    shrinkage: torch.Tensor,
                    selection: torch.Tensor,
                    *,
                    is_deep_update: bool = True,
                    force_permanent: bool = False) -> None:
        """
        Memorize the given segmentation in all memory stores.

        The batch dimension is 1 if flip augmentation is not used.
        image: RGB image, (1/2)*3*H*W
        pix_feat: from the key encoder, (1/2)*_*H*W
        prob: (1/2)*num_objects*H*W, in [0, 1]
        key/shrinkage/selection: for anisotropic l2, (1/2)*_*H*W
        selection can be None if not using long-term memory
        is_deep_update: whether to use deep update (e.g. with the mask encoder)
        force_permanent: whether to force the memory to be permanent
        """
        if prob.shape[1] == 0:
            # nothing to add
            log.warn('Trying to add an empty object mask to memory!')
            return

        if force_permanent:
            as_permanent = 'all'
        else:
            as_permanent = 'first'

        self.memory.initialize_sensory_if_needed(key, self.object_manager.all_obj_ids)
        if (self.trt_mask_decoder is not None
                and hasattr(self.trt_mask_decoder, 'warmup')
                and not self.flip_aug):
            self.trt_mask_decoder.warmup(len(self.object_manager.all_obj_ids))
        with self.profiler.section('encode_mask'):
            msk_value, sensory, obj_value, _ = self.network.encode_mask(
                image,
                pix_feat,
                self.memory.get_sensory(self.object_manager.all_obj_ids),
                prob,
                deep_update=is_deep_update,
                chunk_size=self.chunk_size,
                need_weights=self.save_aux,
                profiler=self.profiler)
        with self.profiler.section('memory_store'):
            self.memory.add_memory(key,
                                   shrinkage,
                                   msk_value,
                                   obj_value,
                                   self.object_manager.all_obj_ids,
                                   selection=selection,
                                   as_permanent=as_permanent)
        self.last_mem_ti = self.curr_ti
        if is_deep_update:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)

    def _segment(self,
                 key: torch.Tensor,
                 selection: torch.Tensor,
                 pix_feat: torch.Tensor,
                 ms_features: Iterable[torch.Tensor],
                 update_sensory: bool = True) -> torch.Tensor:
        """
        Produce a segmentation using the given features and the memory

        The batch dimension is 1 if flip augmentation is not used.
        key/selection: for anisotropic l2: (1/2) * _ * H * W
        pix_feat: from the key encoder, (1/2) * _ * H * W
        ms_features: an iterable of multiscale features from the encoder, each is (1/2)*_*H*W
                      with strides 16, 8, and 4 respectively
        update_sensory: whether to update the sensory memory

        Returns: (num_objects+1)*H*W normalized probability; the first channel is the background
        """
        bs = key.shape[0]
        if self.flip_aug:
            assert bs == 2
        else:
            assert bs == 1

        if not self.memory.engaged:
            log.warn('Trying to segment without any memory!')
            return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                               device=key.device,
                               dtype=key.dtype)

        # Propagation clears sensory memory between runs; if a partial mask forces
        # segmentation before the next add-memory pass, restore zero sensory state
        # for the already-tracked objects so memory readout can proceed.
        self.memory.initialize_sensory_if_needed(key, self.object_manager.all_obj_ids)

        self.profiler.mark('mem_read')

        # Build TRT readout callback if available
        readout_fn = None
        if (self.trt_readout is not None
                and not self.flip_aug and key.shape[0] == 1):
            _trt = self.trt_readout

            def readout_fn(pf, vis, sens, lm, obj_mem):
                lm_ds = F.avg_pool2d(lm, 16, 16)
                return _trt(pf, vis, sens, lm_ds, obj_mem)

        memory_readout = self.memory.read(
            pix_feat, key, selection, self.last_mask, self.network,
            readout_fn=readout_fn,
            profiler=self.profiler)
        with self.profiler.section('memory_realize'):
            memory_readout = self.object_manager.realize_dict(memory_readout)
        self.profiler.mark('mask_decode')
        current_sensory = self.memory.get_sensory(self.object_manager.all_obj_ids)
        trt_mask_decoder = self._resolve_trt_mask_decoder(memory_readout.shape[1])
        if (trt_mask_decoder is not None
                and not self.flip_aug and memory_readout.shape[0] == 1):
            # TRT mask decoder: run decoder + sensory update in one engine call
            with self.profiler.section('trt_mask_decode'):
                new_sensory, logits = trt_mask_decoder(
                    ms_features[1],   # f8
                    ms_features[2],   # f4
                    memory_readout,
                    current_sensory,
                    profiler=self.profiler,
                )
            # Post-processing (same as CUTIE.segment)
            with self.profiler.section('trt_mask_postprocess'):
                prob = torch.sigmoid(logits)
                logits_agg = aggregate(prob, dim=1)
                logits_agg = F.interpolate(logits_agg, scale_factor=4,
                                           mode='bilinear', align_corners=False)
                pred_prob_with_bg = F.softmax(logits_agg, dim=1)[0]
            if update_sensory:
                self.memory.update_sensory(new_sensory, self.object_manager.all_obj_ids)
        else:
            sensory, _, pred_prob_with_bg = self.network.segment(
                ms_features, memory_readout, current_sensory,
                chunk_size=self.chunk_size, update_sensory=update_sensory)
            # remove batch dim
            if self.flip_aug:
                pred_prob_with_bg = (pred_prob_with_bg[0] +
                                     torch.flip(pred_prob_with_bg[1], dims=[-1])) / 2
            else:
                pred_prob_with_bg = pred_prob_with_bg[0]
            if update_sensory:
                self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)
        return pred_prob_with_bg

    def step(self,
             image: torch.Tensor,
             mask: Optional[torch.Tensor] = None,
             objects: Optional[List[int]] = None,
             *,
             idx_mask: bool = True,
             end: bool = False,
             delete_buffer: bool = True,
             force_permanent: bool = False) -> torch.Tensor:
        """
        Take a step with a new incoming image.
        If there is an incoming mask with new objects, we will memorize them.
        If there is no incoming mask, we will segment the image using the memory.
        In both cases, we will update the memory and return a segmentation.

        image: 3*H*W
        mask: H*W (if idx mask) or len(objects)*H*W or None
        objects: list of object ids that are valid in the mask Tensor.
                The ids themselves do not need to be consecutive/in order, but they need to be 
                in the same position in the list as the corresponding mask
                in the tensor in non-idx-mask mode.
                objects is ignored if the mask is None. 
                If idx_mask is False and objects is None, we sequentially infer the object ids.
        idx_mask: if True, mask is expected to contain an object id at every pixel.
                  If False, mask should have multiple channels with each channel representing one object.
        end: if we are at the end of the sequence, we do not need to update memory
            if unsure just set it to False 
        delete_buffer: whether to delete the image feature buffer after this step
        force_permanent: the memory recorded this frame will be added to the permanent memory
        """
        if objects is None and mask is not None:
            assert not idx_mask
            # One-hot probability masks often include every configured object
            # channel even when most are empty. Drop empty channels here so the
            # internal active-object set matches the actual mask content, while
            # keeping currently tracked objects so explicit zeroed channels are
            # preserved as removals rather than discarded.
            objects, mask = _infer_nonempty_prob_mask_objects(
                mask, preserve_objects=self.object_manager.all_obj_ids)

        # resize input if needed -- currently only used for the GUI
        resize_needed = False
        if self.max_internal_size > 0:
            h, w = image.shape[-2:]
            min_side = min(h, w)
            if min_side > self.max_internal_size:
                resize_needed = True
                if self.torch_rt is not None:
                    image = self.torch_rt.resize_image_down(image, self.max_internal_size)
                    if mask is not None:
                        if idx_mask:
                            mask = self.torch_rt.resize_mask_down(mask, self.max_internal_size)
                        else:
                            mask = self.torch_rt.resize_prob_down(mask, self.max_internal_size)
                else:
                    new_h = int(h / min_side * self.max_internal_size)
                    new_w = int(w / min_side * self.max_internal_size)
                    image = F.interpolate(image.unsqueeze(0),
                                          size=(new_h, new_w),
                                          mode='bilinear',
                                          align_corners=False)[0]
                    if mask is not None:
                        if idx_mask:
                            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                                 size=(new_h, new_w),
                                                 mode='nearest-exact')[0, 0].long()
                        else:
                            mask = F.interpolate(mask.unsqueeze(0),
                                                 size=(new_h, new_w),
                                                 mode='bilinear',
                                                 align_corners=False)[0]

        self.curr_ti += 1
        self.profiler.mark('start')

        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension
        if self.flip_aug:
            image = torch.cat([image, torch.flip(image, dims=[-1])], dim=0)

        # whether to update the working memory
        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or
                        (mask is not None)) and (not end)
        # segment when there is no input mask or when the input mask is incomplete
        need_segment = (mask is None) or (self.object_manager.num_obj > 0
                                          and not self.object_manager.has_all(objects))
        update_sensory = ((self.curr_ti - self.last_mem_ti) in self.stagger_ti) and (not end)

        # encoding the image
        self.profiler.mark('encode')
        ms_feat, pix_feat = self.image_feature_store.get_features(self.curr_ti, image)
        key, shrinkage, selection = self.image_feature_store.get_key(self.curr_ti, image)

        # segmentation from memory if needed
        self.profiler.mark('segment')
        if need_segment:
            pred_prob_with_bg = self._segment(key,
                                              selection,
                                              pix_feat,
                                              ms_feat,
                                              update_sensory=update_sensory)

        # use the input mask if provided
        if mask is not None:
            # inform the manager of the new objects, and get a list of temporary id
            # temporary ids -- indicates the position of objects in the tensor
            # (starts with 1 due to the background channel)
            corresponding_tmp_ids, _ = self.object_manager.add_new_objects(objects)

            mask, _ = pad_divide_by(mask, 16)
            masks_by_tmp_id = _masks_by_tmp_id(
                mask, objects, corresponding_tmp_ids, idx_mask=idx_mask)
            if need_segment:
                # merge predicted mask with the incomplete input mask
                pred_prob_no_bg = pred_prob_with_bg[1:]
                # use the mutual exclusivity of segmentation
                if idx_mask:
                    pred_prob_no_bg[:, mask > 0] = 0
                else:
                    pred_prob_no_bg[:, mask.amax(dim=0) > 0.5] = 0

                new_masks = {}
                for tmp_id in sorted(masks_by_tmp_id):
                    this_mask = masks_by_tmp_id[tmp_id].type_as(pred_prob_no_bg)
                    if tmp_id > pred_prob_no_bg.shape[0]:
                        new_masks[tmp_id] = this_mask
                    else:
                        # +1 for padding the background channel
                        pred_prob_no_bg[tmp_id - 1] = this_mask
                if new_masks:
                    appended_masks = [new_masks[tmp_id].unsqueeze(0) for tmp_id in sorted(new_masks)]
                    mask = torch.cat([pred_prob_no_bg, *appended_masks], dim=0)
                else:
                    mask = pred_prob_no_bg
            elif idx_mask:
                # simply convert cls to one-hot representation
                if len(objects) == 0:
                    if delete_buffer:
                        self.image_feature_store.delete(self.curr_ti)
                    log.warn('Trying to insert an empty mask as memory!')
                    return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                                       device=key.device,
                                       dtype=key.dtype)
                zero_mask = mask.new_zeros(mask.shape[-2:])
                mask = torch.stack([masks_by_tmp_id.get(tmp_id, zero_mask)
                                    for tmp_id in range(1, self.object_manager.num_obj + 1)],
                                   dim=0)
            else:
                zero_mask = mask.new_zeros(mask.shape[-2:])
                mask = torch.stack([masks_by_tmp_id.get(tmp_id, zero_mask)
                                    for tmp_id in range(1, self.object_manager.num_obj + 1)],
                                   dim=0)
            pred_prob_with_bg = aggregate(mask, dim=0)
            pred_prob_with_bg = torch.softmax(pred_prob_with_bg, dim=0)

        self.last_mask = pred_prob_with_bg[1:].unsqueeze(0)
        if self.flip_aug:
            self.last_mask = torch.cat(
                [self.last_mask, torch.flip(self.last_mask, dims=[-1])], dim=0)

        # save as memory if needed
        self.profiler.mark('add_mem')
        if is_mem_frame or force_permanent:
            self._add_memory(image,
                             pix_feat,
                             self.last_mask,
                             key,
                             shrinkage,
                             selection,
                             force_permanent=force_permanent)

        if delete_buffer:
            self.image_feature_store.delete(self.curr_ti)

        self.profiler.mark('resize_up')
        output_prob = unpad(pred_prob_with_bg, self.pad)
        if resize_needed:
            # restore output to the original size
            if self.torch_rt is not None:
                output_prob = self.torch_rt.resize_prob_up(output_prob, self.max_internal_size)
            else:
                output_prob = F.interpolate(output_prob.unsqueeze(0),
                                            size=(h, w),
                                            mode='bilinear',
                                            align_corners=False)[0]

        self.profiler.mark('done')
        self.profiler.finish_frame()
        return output_prob

    def delete_objects(self, objects: List[int]) -> None:
        """
        Delete the given objects from the memory.
        """
        self.object_manager.delete_objects(objects)
        self.memory.purge_except(self.object_manager.all_obj_ids)

    def output_prob_to_mask(self, output_prob: torch.Tensor) -> torch.Tensor:
        mask = torch.argmax(output_prob, dim=0)

        # index in tensor != object id -- remap the ids here
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in self.object_manager.tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id

        return new_mask
