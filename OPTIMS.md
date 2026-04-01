# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000000.mp4` (3600 frames, 1080p source)
GPU: Blackwell (RTX PRO 6000), PyTorch 2.11 + CUDA 13.0, TensorRT 10.15, AMP FP16

## Current state: Native TensorRT engines

### Per-frame breakdown at 1080p (19 objects, `--internal-size 1080`)

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.7 | 1.6% | **TRT engine** (FP16, static shapes) |
| Memory readout | similarity + sparse_readout + pixel_fusion + object_transformer | 65 | 63% | Sparse top-k readout |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 23 | 22% | **TRT engine** (FP16, static shapes, NO=19) |
| Add memory | encode_mask (ResNet18) + memory store | 12 | 12% | PyTorch |
| **Total** | | **~102** | | **~9.8 FPS** |

### Improvement over baseline at 1080p

| Stage | Baseline | Current | Speedup |
|-------|---------|---------|---------|
| Encoder | ~25 ms | 1.7 ms (TRT) | **15x** |
| Memory readout | ~71 ms | ~65 ms | 1.1x |
| Mask decoder | ~55 ms | 23 ms (TRT) | **2.4x** |
| Add memory | ~12 ms | ~12 ms | — |
| **Total** | **163 ms (6.1 FPS)** | **102 ms (9.8 FPS)** | **1.6x** |

## TensorRT engines

Two native TRT engines built at startup (ONNX export + TRT autotuner), cached to `~/.cache/atlas-trt/`. First build takes 10-30s; subsequent loads are instant. Cache key includes resolution, GPU, TRT version, FP16 flag, and weights mtime.

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
- 8 GB TRT workspace required for 1080p x 19 objects

## Other optimizations

- **Sparse memory readout**: `do_softmax_sparse` + `sparse_readout` — avoids creating a dense (N x HW) affinity matrix and doing a full BMM where 99.7% of entries are zero (top_k=30 out of N=10000+ tokens). Uses gather-based readout on just the top-k indices.
- **Step profiler**: `--profile` flag prints per-stage CUDA-timed breakdown every 50 frames.

## TRT readout (pixel_fuser + object_transformer) — investigated, not viable

An ONNX wrapper (`_ReadoutPipelineONNXWrapper`) was implemented with:
- Manual multi-head attention (explicit F.linear + matmul + softmax) replacing nn.MultiheadAttention
- Float additive attention masks (masked_fill with -inf) replacing boolean masks
- Precomputed positional encoding
- Static-shape _aux_mask replacing torch.where

**Result**: Wrapper verified correct in PyTorch (diff max=0.025 vs original). However, TRT engine consistently produces incorrect output (silent corruption, not NaN after FP16 fix). Root cause is in TRT 10.15's ONNX graph handling — not cache, precision, I/O binding, or wrapper logic. The wrapper code is kept in `trt_engine.py` for future TRT versions.

## Historical: torch.compile + FP8 benchmarks (pre-TRT)

| Internal Size | Optimization | FPS | ms/frame | Speedup |
|---------------|-------------|-----|----------|---------|
| 480 (default) | None | 24.0 | 41.7 | baseline |
| 480 (default) | torch.compile | 22.6 | 44.3 | 0.94x |
| 1080 | None | 6.1 | 162.8 | baseline |
| 1080 | torch.compile + FP8 | 6.3 | 158.0 | 1.03x |

## What would help next

1. **Config tuning**: `top_k` (30→15), `max_num_tokens` (10000→5000), `mem_every` (5→10) — trades quality for speed on the 65ms memory readout.
2. **TRT for encode_mask** (12ms): ResNet18 + group convolutions, same export pattern as mask decoder.
3. **Future TRT versions**: retry readout engine — the wrapper code is ready, just needs TRT to handle the graph correctly.
4. **Lower resolution**: `--internal-size 720` cuts total to ~50ms (~20 FPS).

## Scaling behavior

| Internal Size | Baseline | With TRT | FPS |
|---------------|---------|----------|-----|
| 480 | 41.7 ms | ~25 ms | ~40 |
| 720 | 74.7 ms | ~50 ms (est) | ~20 |
| 1080 | 162.8 ms | ~102 ms | ~9.8 |
