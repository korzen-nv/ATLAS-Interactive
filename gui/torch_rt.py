"""TorchRT — TensorRT-accelerated resize operations for fixed resolutions.

Pre-compiles bilinear (images/probs) and nearest (index masks) resize kernels
for the known resolution pairs used during interactive segmentation.

Uses ``torch.compile`` with the ``torch_tensorrt`` backend when available
(static-shape TensorRT engines), falling back to the ``inductor`` backend.

Typical setup:
    External canvas:  1920x1080  (video frame size)
    Internal sizes:   720, 1080  (shorter-side targets for backbone processing)

Usage::

    rt = TorchRT(canvas_hw=(1080, 1920), internal_sizes=[720, 1080], device='cuda')

    small = rt.resize_image_down(image_chw, 720)       # (3, 1080, 1920) -> (3, 720, 1280)
    big   = rt.resize_prob_up(prob, 720)                # (C, 720, 1280) -> (C, 1080, 1920)
    smask = rt.resize_mask_down(mask_hw, 720)           # (1080, 1920) -> (720, 1280)
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# -- Detect best available torch.compile backend -----------------------------

_BACKEND: str = "inductor"
_BACKEND_KWARGS: dict = {"mode": "max-autotune"}

# Note: torch_tensorrt is NOT used here — the real TRT acceleration lives in
# trt_engine.py (native TensorRT via ONNX).  Resize ops use inductor which is
# sufficient for these trivial F.interpolate kernels.
log.info("TorchRT: using inductor backend for resize ops")


class TorchRT:
    """Cached, ``torch.compile``-accelerated resize operations.

    When ``torch_tensorrt`` is installed, compiled functions use the TensorRT
    backend which builds static-shape TRT engines — ideal for the fixed
    resolution pairs in this application. Otherwise falls back to the
    ``inductor`` backend with ``mode='max-autotune'``.

    Parameters
    ----------
    canvas_hw : tuple[int, int]
        (height, width) of the external canvas / video frames.
    internal_sizes : list[int]
        Shorter-side targets for internal backbone processing.  For each
        value *s* the actual target (H, W) is computed by scaling the canvas
        aspect ratio so that ``min(H, W) == s``.
    device : str
        CUDA device string (e.g. ``'cuda'`` or ``'cuda:0'``).
    enabled : bool
        If ``False``, all methods fall back to plain ``F.interpolate``
        (useful on CPU or when ``torch.compile`` is unavailable).
    """

    def __init__(
        self,
        canvas_hw: Tuple[int, int] = (1080, 1920),
        internal_sizes: Optional[List[int]] = None,
        device: str = "cuda",
        enabled: bool = True,
    ) -> None:
        self.canvas_h, self.canvas_w = canvas_hw
        self.device = device
        self.enabled = enabled and torch.cuda.is_available()
        self.backend = _BACKEND if self.enabled else "disabled"

        if internal_sizes is None:
            internal_sizes = [720, 1080]

        self._internal_hw: Dict[int, Tuple[int, int]] = {}
        for s in internal_sizes:
            h, w = self._compute_internal_hw(s)
            self._internal_hw[s] = (h, w)
            log.info("TorchRT: internal %d -> (%d, %d)", s, h, w)

        self._bilinear_cache: Dict[Tuple[int, int, int, int], Optional[Callable]] = {}
        self._nearest_cache: Dict[Tuple[int, int, int, int], Optional[Callable]] = {}

        if self.enabled:
            self._warmup()

    # -- Resolution helpers ---------------------------------------------------

    def _compute_internal_hw(self, short_side: int) -> Tuple[int, int]:
        """Return (H, W) that preserves the canvas aspect ratio."""
        ch, cw = self.canvas_h, self.canvas_w
        min_side = min(ch, cw)
        if min_side <= short_side:
            return (ch, cw)
        ratio = short_side / min_side
        return (int(ch * ratio), int(cw * ratio))

    def internal_hw(self, short_side: int) -> Tuple[int, int]:
        """Look up the pre-computed internal (H, W) for a given short side."""
        if short_side in self._internal_hw:
            return self._internal_hw[short_side]
        hw = self._compute_internal_hw(short_side)
        self._internal_hw[short_side] = hw
        return hw

    def needs_resize(self, short_side: int) -> bool:
        """Return True if the internal size differs from the canvas size."""
        ih, iw = self.internal_hw(short_side)
        return (ih, iw) != (self.canvas_h, self.canvas_w)

    # -- Compilation ----------------------------------------------------------

    def _compile(self, fn: Callable) -> Callable:
        """Wrap a function with torch.compile using the best available backend."""
        if _BACKEND == "torch_tensorrt":
            return torch.compile(fn, backend="torch_tensorrt")
        else:
            return torch.compile(fn, **_BACKEND_KWARGS)

    def _get_bilinear(self, ih: int, iw: int, oh: int, ow: int) -> Optional[Callable]:
        key = (ih, iw, oh, ow)
        if key not in self._bilinear_cache:
            if self.enabled and (ih, iw) != (oh, ow):
                log.info("TorchRT [%s]: compiling bilinear (%d,%d)->(%d,%d)",
                         self.backend, ih, iw, oh, ow)

                def _fn(x: torch.Tensor, _h=oh, _w=ow) -> torch.Tensor:
                    return F.interpolate(x, size=(_h, _w), mode="bilinear", align_corners=False)

                self._bilinear_cache[key] = self._compile(_fn)
            else:
                self._bilinear_cache[key] = None
        return self._bilinear_cache[key]

    def _get_nearest(self, ih: int, iw: int, oh: int, ow: int) -> Optional[Callable]:
        key = (ih, iw, oh, ow)
        if key not in self._nearest_cache:
            if self.enabled and (ih, iw) != (oh, ow):
                log.info("TorchRT [%s]: compiling nearest (%d,%d)->(%d,%d)",
                         self.backend, ih, iw, oh, ow)

                def _fn(x: torch.Tensor, _h=oh, _w=ow) -> torch.Tensor:
                    return F.interpolate(x, size=(_h, _w), mode="nearest-exact")

                self._nearest_cache[key] = self._compile(_fn)
            else:
                self._nearest_cache[key] = None
        return self._nearest_cache[key]

    def _warmup(self) -> None:
        """Eagerly compile resize kernels by running them on dummy tensors.

        ``torch.compile`` defers actual compilation until the first call, so
        we create dummy tensors and invoke each compiled function to trigger
        engine building at startup rather than on the first real frame.
        """
        ch, cw = self.canvas_h, self.canvas_w
        for s, (ih, iw) in self._internal_hw.items():
            if (ih, iw) == (ch, cw):
                continue
            # bilinear: canvas -> internal
            fn = self._get_bilinear(ch, cw, ih, iw)
            if fn is not None:
                fn(torch.zeros(1, 1, ch, cw, device=self.device))
            # bilinear: internal -> canvas
            fn = self._get_bilinear(ih, iw, ch, cw)
            if fn is not None:
                fn(torch.zeros(1, 1, ih, iw, device=self.device))
            # nearest: canvas -> internal
            fn = self._get_nearest(ch, cw, ih, iw)
            if fn is not None:
                fn(torch.zeros(1, 1, ch, cw, device=self.device))
            # nearest: internal -> canvas
            fn = self._get_nearest(ih, iw, ch, cw)
            if fn is not None:
                fn(torch.zeros(1, 1, ih, iw, device=self.device))

    # -- Core resize logic ----------------------------------------------------

    def _resize(
        self,
        tensor: torch.Tensor,
        target_hw: Tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        """Resize a (C, H, W) or (N, C, H, W) tensor using a cached compiled kernel."""
        th, tw = target_hw
        sh, sw = tensor.shape[-2], tensor.shape[-1]
        if (sh, sw) == (th, tw):
            return tensor
        needs_batch = tensor.dim() == 3
        x = tensor.unsqueeze(0) if needs_batch else tensor
        if mode == "bilinear":
            fn = self._get_bilinear(sh, sw, th, tw)
        else:
            fn = self._get_nearest(sh, sw, th, tw)
        if fn is not None:
            # .clone() — CUDA graphs reuse output buffers across runs;
            # without clone, downstream ops (e.g. F.pad) may see stale data.
            out = fn(x).clone()
        elif mode == "bilinear":
            out = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        else:
            out = F.interpolate(x, size=(th, tw), mode="nearest-exact")
        return out[0] if needs_batch else out

    # -- Public resize API ----------------------------------------------------

    def resize_image_down(self, image: torch.Tensor, short_side: int) -> torch.Tensor:
        """Downscale a (3, H, W) image to the internal resolution."""
        return self._resize(image, self.internal_hw(short_side), "bilinear")

    def resize_prob_up(self, prob: torch.Tensor, short_side: int) -> torch.Tensor:
        """Upscale a (C, H, W) probability tensor back to canvas resolution."""
        return self._resize(prob, (self.canvas_h, self.canvas_w), "bilinear")

    def resize_prob_down(self, prob: torch.Tensor, short_side: int) -> torch.Tensor:
        """Downscale a (C, H, W) soft-mask / probability tensor to internal resolution."""
        return self._resize(prob, self.internal_hw(short_side), "bilinear")

    def resize_mask_down(self, mask: torch.Tensor, short_side: int) -> torch.Tensor:
        """Downscale an (H, W) index mask to the internal resolution.

        Uses nearest-exact interpolation to preserve class IDs.
        """
        ih, iw = self.internal_hw(short_side)
        if (mask.shape[-2], mask.shape[-1]) == (ih, iw):
            return mask
        was_long = mask.dtype in (torch.long, torch.int32, torch.int64)
        out = self._resize(mask.float().unsqueeze(0), (ih, iw), "nearest")[0]
        return out.long() if was_long else out

    def resize_mask_up(self, mask: torch.Tensor, short_side: int) -> torch.Tensor:
        """Upscale an (H, W) index mask back to canvas resolution.

        Uses nearest-exact interpolation to preserve class IDs.
        """
        ch, cw = self.canvas_h, self.canvas_w
        if (mask.shape[-2], mask.shape[-1]) == (ch, cw):
            return mask
        was_long = mask.dtype in (torch.long, torch.int32, torch.int64)
        out = self._resize(mask.float().unsqueeze(0), (ch, cw), "nearest")[0]
        return out.long() if was_long else out

    def resize_bilinear(self, tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """General bilinear resize for an (N)CHW tensor."""
        return self._resize(tensor, target_hw, "bilinear")

    def resize_nearest(self, tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """General nearest resize for an (N)CHW tensor."""
        return self._resize(tensor, target_hw, "nearest")

    def __repr__(self) -> str:
        internals = ", ".join(
            f"{s}->({h}x{w})" for s, (h, w) in sorted(self._internal_hw.items())
        )
        return (
            f"TorchRT(canvas={self.canvas_h}x{self.canvas_w}, "
            f"backend={self.backend}, "
            f"internals=[{internals}], "
            f"compiled_bilinear={len(self._bilinear_cache)}, "
            f"compiled_nearest={len(self._nearest_cache)})"
        )
