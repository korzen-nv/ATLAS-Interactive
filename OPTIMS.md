# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000000.mp4` (3600 frames, 1080p source)
GPU: Blackwell (RTX PRO 6000), PyTorch 2.11 + CUDA 13.0, AMP FP16 always on

## Current state: Native TensorRT engines

### Per-frame breakdown at 1080p (19 objects, `--internal-size 1080`)

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.7 | 1.7% | **TRT engine** (FP16, static shapes) |
| Memory readout | similarity + sparse_readout + pixel_fusion + object_transformer | 65 | 64% | Sparse top-k readout; pixel_fusion + obj_transformer still PyTorch |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 23 | 22% | **TRT engine** (FP16, static shapes, NO=19) |
| Add memory | encode_mask (ResNet18) + memory store | 12 | 12% | PyTorch (not on hot path every frame) |
| Resize + pad | bilinear up/down | <0.3 | <0.3% | torch.compile (inductor) |
| **Total** | | **~102** | | **~9.8 FPS** |

### Improvement over baseline at 1080p

| Stage | Baseline | Current | Speedup |
|-------|---------|---------|---------|
| Encoder | ~25 ms | 1.7 ms (TRT) | **15x** |
| Memory readout | ~71 ms | ~65 ms | 1.1x |
| Mask decoder | ~55 ms | 23 ms (TRT) | **2.4x** |
| Add memory | ~12 ms | ~12 ms | — |
| **Total** | **163 ms (6.1 FPS)** | **102 ms (9.8 FPS)** | **1.6x** |

### Per-frame breakdown at 480p (19 objects, default `--internal-size 480`)

| Stage | ms/frame (approx) | Note |
|-------|--------------------|------|
| Encoder (TRT) | ~1.7 | Same engine, smaller spatial dims |
| Memory readout | ~15 | ~5x fewer spatial tokens than 1080 |
| Mask decoder (TRT) | ~5 | ~5x less spatial area |
| Total | ~25 | ~40 FPS |

## TensorRT engines

Two native TRT engines are built at startup (ONNX export + TRT autotuner) and cached to `~/.cache/atlas-trt/`. First build takes 10-30s; subsequent loads are instant.

**Encoder engine** (`TRTEncoder`):
- Combines: PixelEncoder (ResNet50 layers 1-3) + pix_feat_proj (1x1 Conv) + KeyProjection (3 Conv2d + pow + sigmoid)
- Input: `(1, 3, H, W)` raw image (normalisation baked in)
- 7 outputs: f16, f8, f4, pix_feat, key, shrinkage, selection
- Conv-BN-ReLU fusion, FP16 kernels throughout

**Mask decoder engine** (`TRTMaskDecoder`):
- Combines: DecoderFeatureProcessor + MaskUpsampleBlock (x2) + prediction Conv + SensoryUpdater GRU
- Inputs: f8_raw, f4_raw, memory_readout, sensory (fixed `num_objects` from config)
- 2 outputs: new_sensory, logits
- Handles fewer active objects via zero-padding + output slicing
- Uses avg_pool2d instead of F.interpolate(mode='area') for ONNX compatibility
- 8 GB TRT workspace required for 1080p × 19 objects

### TRT readout (pixel_fuser + object_transformer) — not yet working

An ONNX wrapper for the pixel_fusion + object_transformer pipeline was implemented and verified correct in PyTorch (diff < 0.025 vs original). However, the ONNX → TRT conversion produces NaN output — likely due to `nn.MultiheadAttention` with boolean/float-inf attention masks not surviving the ONNX export correctly. The wrapper code (`_ReadoutPipelineONNXWrapper` in `trt_engine.py`) is kept for future work.

## Other optimizations applied

- **Sparse memory readout**: `do_softmax_sparse` + `sparse_readout` — avoids creating a dense (N x HW) affinity matrix and doing a full BMM where 99.7% of entries are zero (top_k=30 out of N=10000+ tokens). Uses gather-based readout on just the top-k indices.

## Historical: torch.compile + FP8 benchmarks (pre-TRT)

| Internal Size | Optimization | FPS | ms/frame | Speedup |
|---------------|-------------|-----|----------|---------|
| 480 (default) | None | 24.0 | 41.7 | baseline |
| 480 (default) | torch.compile | 22.6 | 44.3 | 0.94x (slower) |
| 720 | None | 13.4 | 74.7 | baseline |
| 1080 | None | 6.1 | 162.8 | baseline |
| 1080 | torch.compile | 6.3 | 159.8 | 1.02x |
| 1080 | FP8 weight-only | 6.3 | 159.3 | 1.02x |
| 1080 | torch.compile + FP8 | 6.3 | 158.0 | 1.03x |

**Why torch.compile/FP8 gains were negligible**: only ~30% of compute (encoder) was compiled; cuDNN ResNet convolutions already near-optimal; FP8 weight-only saves little for conv layers dominated by activation memory traffic.

## What would help next

1. **TRT for pixel_fusion + object_transformer** (65ms = 64% of frame time at 1080): the wrapper logic is correct (verified in PyTorch) but ONNX export of `nn.MultiheadAttention` with masked attention produces NaN in TRT. Needs either: (a) manual attention implementation avoiding MHA for ONNX, or (b) torch_tensorrt AOT compilation which handles MHA natively.
2. **Config tuning**: reduce `top_k` (30 → 15), `max_num_tokens` (10000 → 5000), increase `mem_every` — trades quality for speed on memory readout.
3. **TRT for encode_mask** (12ms = 12%): ResNet18 + group convolutions. Same pattern as mask decoder export.

## Scaling behavior

| Internal Size | ms/frame (baseline) | ms/frame (TRT) | FPS (TRT) |
|---------------|--------------------|--------------------|-----------|
| 480 | 41.7 | ~25 | ~40 |
| 720 | 74.7 | ~50 (est) | ~20 |
| 1080 | 162.8 | ~102 | ~9.8 |

Roughly quadratic in spatial dimensions (1080/480 = 2.25x per side, 2.25^2 = 5x theoretical).
