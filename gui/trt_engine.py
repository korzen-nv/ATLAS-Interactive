"""Native TensorRT acceleration for the CUTIE encoder pipeline.

Exports pixel_encoder + pix_feat_proj + key_proj as a single static-shape
TensorRT engine with FP16.  The engine is built once per (resolution, GPU,
model weights, TRT version) and cached to disk (~/.cache/atlas-trt/).

Typical speedup: 2-3x on the encoder vs eager PyTorch, from Conv-BN-ReLU
fusion, FP16 compute kernels, and kernel autotuning.

Falls back gracefully when ``tensorrt`` is not installed or build fails.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TensorRT availability
# ---------------------------------------------------------------------------
_TRT_AVAILABLE = False
try:
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True if the ``tensorrt`` Python package is importable."""
    return _TRT_AVAILABLE


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------
OUTPUT_NAMES = ["f16", "f8", "f4", "pix_feat", "key", "shrinkage", "selection"]


class _EncoderONNXWrapper(nn.Module):
    """Wraps pixel_encoder + pix_feat_proj + key_proj for clean ONNX export.

    Bakes in image normalisation and hardcodes ``need_s=True, need_e=True``
    (the inference-time defaults) so the graph has no conditional branches.

    Produces seven outputs:
        f16, f8, f4       — multi-scale pixel features (strides 16, 8, 4)
        pix_feat           — projected pixel features (256-d)
        key, shrinkage, selection — key projection outputs
    """

    def __init__(self, cutie: nn.Module) -> None:
        super().__init__()
        # Pixel encoder (ResNet50 layers 1-3)
        self.pixel_encoder = cutie.pixel_encoder
        # CUTIE's own 1×1 pixel feature projection
        self.pix_feat_proj = cutie.pix_feat_proj
        # KeyProjection sub-layers (unwrapped to avoid bool kwargs in ONNX)
        self.kp_pix_feat_proj = cutie.key_proj.pix_feat_proj
        self.kp_key_proj = cutie.key_proj.key_proj
        self.kp_d_proj = cutie.key_proj.d_proj
        self.kp_e_proj = cutie.key_proj.e_proj
        # Normalisation constants
        self.register_buffer("pixel_mean", cutie.pixel_mean.clone())
        self.register_buffer("pixel_std", cutie.pixel_std.clone())

    def forward(self, image: torch.Tensor):
        x = (image - self.pixel_mean) / self.pixel_std
        f16, f8, f4 = self.pixel_encoder(x)
        pix_feat = self.pix_feat_proj(f16)
        kp_x = self.kp_pix_feat_proj(f16)
        key = self.kp_key_proj(kp_x)
        shrinkage = self.kp_d_proj(kp_x) ** 2 + 1
        selection = torch.sigmoid(self.kp_e_proj(kp_x))
        return f16, f8, f4, pix_feat, key, shrinkage, selection


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------
if _TRT_AVAILABLE:
    _TRT_TO_TORCH = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
    }
else:
    _TRT_TO_TORCH = {}


# ---------------------------------------------------------------------------
# TRT encoder runtime
# ---------------------------------------------------------------------------
class TRTEncoder:
    """Runs the CUTIE encoder pipeline on a native TensorRT engine.

    Accepts a ``(1, 3, H, W)`` float32 image and returns seven tensors
    matching the outputs of ``CUTIE.encode_image`` + ``CUTIE.transform_key``.
    """

    def __init__(self, engine_bytes: bytes, device: str = "cuda") -> None:
        if not _TRT_AVAILABLE:
            raise RuntimeError("tensorrt Python package is not installed")

        self._device = torch.device(device)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with torch.cuda.device(self._device):
            self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()

        # Discover I/O tensor specs from the engine
        self._input_name: str = ""
        self._output_specs: dict[str, Tuple[tuple, torch.dtype]] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                shape = tuple(self._engine.get_tensor_shape(name))
                dtype = _TRT_TO_TORCH.get(
                    self._engine.get_tensor_dtype(name), torch.float32
                )
                self._output_specs[name] = (shape, dtype)

        log.info(
            "TRTEncoder: ready (%d outputs, input=%s)",
            len(self._output_specs),
            self._input_name,
        )

    # ------------------------------------------------------------------ call
    def __call__(self, image: torch.Tensor):
        """Run encoder on a **(1, 3, H, W)** float32 image.

        Returns ``(f16, f8, f4, pix_feat, key, shrinkage, selection)``.
        """
        image = image.contiguous().float()

        # Allocate fresh output tensors each call (CUDA pool makes this cheap)
        # so callers can safely cache the returned tensors.
        outputs: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in self._output_specs.items():
            outputs[name] = torch.empty(shape, dtype=dtype, device=self._device)

        # Bind I/O addresses
        self._context.set_tensor_address(self._input_name, image.data_ptr())
        for name, tensor in outputs.items():
            self._context.set_tensor_address(name, tensor.data_ptr())

        # Execute on the current CUDA stream (naturally ordered with PyTorch)
        stream = torch.cuda.current_stream(self._device)
        ok = self._context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRTEncoder: execute_async_v3 failed")

        return tuple(outputs[name] for name in OUTPUT_NAMES)

    # -------------------------------------------------------------- builder
    @staticmethod
    def build(
        cutie: nn.Module,
        input_hw: Tuple[int, int],
        *,
        weights_path: str = "",
        cache_dir: Optional[str] = None,
        fp16: bool = True,
        device: str = "cuda",
    ) -> Optional[TRTEncoder]:
        """Export ONNX → build TRT engine (or load from disk cache).

        Parameters
        ----------
        cutie : nn.Module
            The **unmodified** CUTIE model (before FP8 / torch.compile).
        input_hw : (int, int)
            Padded internal resolution ``(H, W)``, divisible by 16.
        weights_path : str
            Path to the ``.pth`` file — its mtime enters the cache key so a
            weight change invalidates stale engines.
        cache_dir : str | None
            Directory for engine files.  Defaults to ``~/.cache/atlas-trt``.
        fp16 : bool
            Enable FP16 in the TRT builder (recommended).
        device : str
            CUDA device, e.g. ``'cuda'`` or ``'cuda:0'``.

        Returns
        -------
        TRTEncoder | None
            The ready-to-use engine, or ``None`` on failure.
        """
        if not _TRT_AVAILABLE:
            log.info("TRTEncoder: tensorrt not installed — skipping")
            return None

        h, w = input_hw
        gpu_name = torch.cuda.get_device_name(device).replace(" ", "_")

        # ---- cache key ----
        w_mtime = ""
        if weights_path and os.path.exists(weights_path):
            w_mtime = str(int(os.path.getmtime(weights_path)))
        key_str = f"{h}x{w}_{gpu_name}_trt{trt.__version__}_{w_mtime}"
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

        if cache_dir is None:
            cache_dir = str(Path.home() / ".cache" / "atlas-trt")
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        engine_file = cache_path / f"{key_hash}.engine"

        # ---- try loading cached engine ----
        if engine_file.exists():
            log.info("TRTEncoder: loading cached engine %s", engine_file)
            try:
                return TRTEncoder(engine_file.read_bytes(), device=device)
            except Exception as exc:
                log.warning("TRTEncoder: cache load failed (%s), rebuilding", exc)
                engine_file.unlink(missing_ok=True)

        # ---- build from scratch ----
        log.info(
            "TRTEncoder: building %dx%d on %s FP16=%s (first-time only) ...",
            h,
            w,
            gpu_name,
            fp16,
        )
        t0 = time.time()

        try:
            return _build_engine(cutie, h, w, fp16, device, cache_path, key_hash, engine_file)
        except Exception as exc:
            log.warning("TRTEncoder: build failed — %s", exc)
            return None


# ---------------------------------------------------------------------------
# Internal build helpers (kept outside the class to reduce indentation)
# ---------------------------------------------------------------------------
def _cleanup_onnx(onnx_path: Path) -> None:
    """Remove ONNX file and its external data companion (.onnx.data)."""
    onnx_path.unlink(missing_ok=True)
    data_path = onnx_path.with_suffix(".onnx.data")
    data_path.unlink(missing_ok=True)


def _onnx_to_trt(
    wrapper: nn.Module,
    dummy_inputs: tuple,
    input_names: list,
    output_names: list,
    cache_path: Path,
    key_hash: str,
    engine_file: Path,
    fp16: bool,
    *,
    label: str = "TRT",
    workspace_gb: int = 2,
) -> Optional[bytes]:
    """Generic ONNX export → TRT engine build → cache to disk.

    Returns engine bytes on success, ``None`` on failure.
    """
    onnx_path = cache_path / f"{key_hash}.onnx"

    # Step 1: ONNX export
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
        )
    log.info("%s: ONNX exported → %s", label, onnx_path)

    # Step 2: parse ONNX into a TRT network
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)

    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        flag = 0
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(str(onnx_path)):
        for j in range(parser.num_errors):
            log.error("%s ONNX parse error: %s", label, parser.get_error(j))
        _cleanup_onnx(onnx_path)
        return None

    # Step 3: build
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
    )
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    t_build = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        log.error("%s: engine serialisation failed", label)
        _cleanup_onnx(onnx_path)
        return None

    engine_bytes = bytes(serialized)
    engine_file.write_bytes(engine_bytes)
    _cleanup_onnx(onnx_path)

    size_mb = len(engine_bytes) / (1024 * 1024)
    elapsed = time.time() - t_build
    log.info(
        "%s: engine built in %.1fs (%.1f MB) → %s",
        label, elapsed, size_mb, engine_file,
    )
    return engine_bytes


def _build_engine(
    cutie: nn.Module,
    h: int,
    w: int,
    fp16: bool,
    device: str,
    cache_path: Path,
    key_hash: str,
    engine_file: Path,
) -> Optional[TRTEncoder]:
    """Build encoder TRT engine (uses shared _onnx_to_trt)."""
    wrapper = _EncoderONNXWrapper(cutie).eval().to(device)
    dummy = (torch.randn(1, 3, h, w, device=device),)

    engine_bytes = _onnx_to_trt(
        wrapper, dummy, ["image"], OUTPUT_NAMES,
        cache_path, key_hash, engine_file, fp16,
        label="TRTEncoder",
    )
    if engine_bytes is None:
        return None
    return TRTEncoder(engine_bytes, device=device)


# ===================================================================
# Mask Decoder TRT
# ===================================================================

MASK_DEC_INPUT_NAMES = ["f8_raw", "f4_raw", "memory_readout", "sensory"]
MASK_DEC_OUTPUT_NAMES = ["new_sensory", "logits"]


class _MaskDecoderONNXWrapper(nn.Module):
    """Wraps MaskDecoder for ONNX export with fixed num_objects and spatial dims.

    Combines decoder_feat_proc + upsample blocks + pred + sensory_update.
    Replaces ``F.interpolate(mode='area')`` with ``avg_pool2d`` for ONNX/TRT
    compatibility, and inlines the GRU update to avoid ``autocast`` context.
    """

    def __init__(self, cutie: nn.Module, num_objects: int) -> None:
        super().__init__()
        dec = cutie.mask_decoder
        self.num_objects = num_objects

        # decoder_feat_proc: project skip features
        self.feat_f8 = dec.decoder_feat_proc.transforms[0]
        self.feat_f4 = dec.decoder_feat_proc.transforms[1]

        # upsampling path
        self.up_16_8 = dec.up_16_8
        self.up_8_4 = dec.up_8_4

        # prediction head
        self.pred = dec.pred

        # sensory updater sub-layers (referenced, not copied)
        self.su_g16 = dec.sensory_update.g16_conv
        self.su_g8 = dec.sensory_update.g8_conv
        self.su_g4 = dec.sensory_update.g4_conv
        self.su_transform = dec.sensory_update.transform

    def forward(
        self,
        f8_raw: torch.Tensor,
        f4_raw: torch.Tensor,
        memory_readout: torch.Tensor,
        sensory: torch.Tensor,
    ):
        """
        f8_raw:          (1, 512, H8, W8)    — raw encoder stride-8 feature
        f4_raw:          (1, 256, H4, W4)    — raw encoder stride-4 feature
        memory_readout:  (1, NO,  256, H16, W16) — from pixel_fusion + obj transformer
        sensory:         (1, NO,  256, H16, W16) — recurrent sensory memory

        Returns:
            new_sensory: (1, NO, 256, H16, W16)
            logits:      (1, NO, H4, W4)
        """
        NO = self.num_objects

        # --- skip feature projection ---
        f8 = self.feat_f8(f8_raw)
        f4 = self.feat_f4(f4_raw)

        # --- upsample path ---
        p16 = memory_readout
        p8 = self.up_16_8(p16, f8)
        p4 = self.up_8_4(p8, f4)

        # --- prediction ---
        logits = self.pred(torch.relu(p4.flatten(0, 1)))  # (NO, 1, H4, W4)

        # --- sensory GRU update ---
        # g16 = su_g16(p16)  — stride 16, no resize
        g16 = self.su_g16(p16)

        # g8 = su_g8(downsample(p8, 0.5))
        # Use avg_pool2d instead of F.interpolate(mode='area') for ONNX compat
        p8_ds = nn.functional.avg_pool2d(p8.flatten(0, 1), 2, 2)
        p8_ds = p8_ds.view(1, NO, p8_ds.shape[1], p8_ds.shape[2], p8_ds.shape[3])
        g8 = self.su_g8(p8_ds)

        # g4 = su_g4(downsample(cat(p4, logits), 0.25))
        p4_cat = torch.cat(
            [p4, logits.view(1, NO, 1, logits.shape[-2], logits.shape[-1])], 2
        )
        p4_ds = nn.functional.avg_pool2d(p4_cat.flatten(0, 1), 4, 4)
        p4_ds = p4_ds.view(1, NO, p4_ds.shape[1], p4_ds.shape[2], p4_ds.shape[3])
        g4 = self.su_g4(p4_ds)

        g = g16 + g8 + g4

        # GRU update (inlined from _recurrent_update, always FP32)
        g = g.float()
        h = sensory.float()
        values = self.su_transform(torch.cat([g, h], dim=2))
        dim_v = values.shape[2] // 3
        forget_gate = torch.sigmoid(values[:, :, :dim_v])
        update_gate = torch.sigmoid(values[:, :, dim_v : dim_v * 2])
        new_value = torch.tanh(values[:, :, dim_v * 2 :])
        new_sensory = forget_gate * h * (1 - update_gate) + update_gate * new_value

        # Reshape logits: (NO, 1, H4, W4) → (1, NO, H4, W4)
        logits = logits.squeeze(1).unsqueeze(0)

        return new_sensory, logits


class TRTMaskDecoder:
    """Runs the CUTIE mask decoder on a native TensorRT engine."""

    def __init__(self, engine_bytes: bytes, device: str = "cuda") -> None:
        if not _TRT_AVAILABLE:
            raise RuntimeError("tensorrt not installed")

        self._device = torch.device(device)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with torch.cuda.device(self._device):
            self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()

        self._input_specs: dict[str, Tuple[tuple, torch.dtype]] = {}
        self._output_specs: dict[str, Tuple[tuple, torch.dtype]] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH.get(
                self._engine.get_tensor_dtype(name), torch.float32
            )
            if mode == trt.TensorIOMode.INPUT:
                self._input_specs[name] = (shape, dtype)
            else:
                self._output_specs[name] = (shape, dtype)

        self.num_objects = self._input_specs["memory_readout"][0][1]
        log.info(
            "TRTMaskDecoder: ready (num_objects=%d, %d inputs, %d outputs)",
            self.num_objects,
            len(self._input_specs),
            len(self._output_specs),
        )

    def __call__(
        self,
        f8_raw: torch.Tensor,
        f4_raw: torch.Tensor,
        memory_readout: torch.Tensor,
        sensory: torch.Tensor,
    ):
        """Run mask decoder.  Inputs may have fewer objects than engine max —
        they will be padded to ``self.num_objects`` automatically.

        Returns ``(new_sensory, logits)`` sliced back to actual object count.
        """
        actual_no = memory_readout.shape[1]
        need_pad = actual_no < self.num_objects

        if need_pad:
            pad_no = self.num_objects - actual_no
            memory_readout = nn.functional.pad(memory_readout, (0, 0, 0, 0, 0, 0, 0, pad_no))
            sensory = nn.functional.pad(sensory, (0, 0, 0, 0, 0, 0, 0, pad_no))

        inputs = {
            "f8_raw": f8_raw.contiguous().float(),
            "f4_raw": f4_raw.contiguous().float(),
            "memory_readout": memory_readout.contiguous().float(),
            "sensory": sensory.contiguous().float(),
        }
        outputs: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in self._output_specs.items():
            outputs[name] = torch.empty(shape, dtype=dtype, device=self._device)

        for name, tensor in inputs.items():
            self._context.set_tensor_address(name, tensor.data_ptr())
        for name, tensor in outputs.items():
            self._context.set_tensor_address(name, tensor.data_ptr())

        stream = torch.cuda.current_stream(self._device)
        ok = self._context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRTMaskDecoder: execute_async_v3 failed")

        new_sensory = outputs["new_sensory"]
        logits = outputs["logits"]

        if need_pad:
            new_sensory = new_sensory[:, :actual_no]
            logits = logits[:, :actual_no]

        return new_sensory, logits

    @staticmethod
    def build(
        cutie: nn.Module,
        input_hw: Tuple[int, int],
        num_objects: int,
        *,
        weights_path: str = "",
        cache_dir: Optional[str] = None,
        fp16: bool = True,
        device: str = "cuda",
    ) -> Optional["TRTMaskDecoder"]:
        """Build (or load cached) TRT engine for the mask decoder."""
        if not _TRT_AVAILABLE:
            log.info("TRTMaskDecoder: tensorrt not installed — skipping")
            return None

        h, w = input_hw
        h16, w16 = h // 16, w // 16
        h8, w8 = h // 8, w // 8
        h4, w4 = h // 4, w // 4
        gpu_name = torch.cuda.get_device_name(device).replace(" ", "_")
        NO = num_objects

        w_mtime = ""
        if weights_path and os.path.exists(weights_path):
            w_mtime = str(int(os.path.getmtime(weights_path)))
        key_str = (
            f"maskdec_{h}x{w}_no{NO}_{gpu_name}_trt{trt.__version__}_{w_mtime}"
        )
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

        if cache_dir is None:
            cache_dir = str(Path.home() / ".cache" / "atlas-trt")
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        engine_file = cache_path / f"{key_hash}.engine"

        if engine_file.exists():
            log.info("TRTMaskDecoder: loading cached %s", engine_file)
            try:
                return TRTMaskDecoder(engine_file.read_bytes(), device=device)
            except Exception as exc:
                log.warning("TRTMaskDecoder: cache load failed (%s), rebuilding", exc)
                engine_file.unlink(missing_ok=True)

        log.info(
            "TRTMaskDecoder: building %dx%d NO=%d on %s (first-time only) ...",
            h, w, NO, gpu_name,
        )

        try:
            wrapper = _MaskDecoderONNXWrapper(cutie, NO).eval().to(device)
            dummy = (
                torch.randn(1, 512, h8, w8, device=device),
                torch.randn(1, 256, h4, w4, device=device),
                torch.randn(1, NO, 256, h16, w16, device=device),
                torch.randn(1, NO, 256, h16, w16, device=device),
            )
            engine_bytes = _onnx_to_trt(
                wrapper,
                dummy,
                MASK_DEC_INPUT_NAMES,
                MASK_DEC_OUTPUT_NAMES,
                cache_path,
                key_hash,
                engine_file,
                fp16,
                label="TRTMaskDecoder",
                workspace_gb=8,
            )
            if engine_bytes is None:
                return None
            return TRTMaskDecoder(engine_bytes, device=device)
        except Exception as exc:
            log.warning("TRTMaskDecoder: build failed — %s", exc)
            return None
