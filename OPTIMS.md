# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000001.mp4` (3600 frames, 1080p source)
GPU: Blackwell (RTX PRO 6000), PyTorch 2.11 + CUDA 13.0, TensorRT 10.15, AMP FP16

## Current state: TRT encoder + TRT mask decoder + optimized readout

### Per-frame breakdown at 1080p (19 objects, `--internal-size 1080`)

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.7 | 1.8% | **TRT engine** (FP16, static shapes) |
| Memory readout | affinity + sparse_readout + pixel_fusion + object_transformer | 59.1 | 61.7% | Streamed sparse top-k + SDPA path |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 22.9 | 23.9% | **TRT engine** (FP16, static shapes, NO=19) |
| Add memory | encode_mask (ResNet18) + memory store | 12.0 | 12.5% | PyTorch |
| **Total** | | **95.8** | | **10.4 FPS** |

### Detailed readout breakdown (from `--profile`)

| Sub-stage | ms/frame | Notes |
|----------|----------|-------|
| `affinity_topk` | 21.50 | Largest remaining hotspot; still chunked PyTorch, not a fused kernel |
| `sparse_readout` | 15.23 | Gather/reduction over top-k values is now a major bottleneck |
| `object_transformer` | 13.08 | Improved with SDPA + lighter mask handling, but still significant |
| `pixel_fusion` | 5.38 | Secondary cost |
| `aux_mask` | 0.39 | Negligible after additive-mask cleanup |

### Improvement over baseline at 1080p

| Stage | Baseline | Current | Speedup |
|-------|---------|---------|---------|
| Encoder | ~25 ms | 1.7 ms (TRT) | **15x** |
| Memory readout | ~71 ms | 59.1 ms | **1.2x** |
| Mask decoder | ~55 ms | 22.9 ms (TRT) | **2.4x** |
| Add memory | ~12 ms | ~12 ms | — |
| **Total** | **163 ms (6.1 FPS)** | **95.8 ms (10.4 FPS)** | **1.7x** |

## TensorRT engines

Two native TRT engines built at startup (ONNX export + TRT autotuner), cached to `~/.cache/atlas-trt/`. First build takes 10-30s; subsequent loads are instant. Cache key includes resolution, GPU, TRT version, FP16 flag, model signature, and weights mtime.

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

- **Streamed sparse affinity**: `sparse_topk_affinity` now chunks over memory tokens and avoids materializing the full dense `(B, N, HW)` similarity tensor in the CUDA fast path.
- **Transformer fast path**: query transformer attention now uses PyTorch SDPA with a broadcast additive mask instead of per-head mask replication.
- **Cached positional encoding**: spatial PE is cached by shape/layout instead of batch/object count.
- **Step profiler**: `--profile` now prints both coarse stage timings and detailed readout sub-stages (`affinity_topk`, `sparse_readout`, `pixel_fusion`, `object_transformer`, `aux_mask`).

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

## Findings

1. **Readout is still the bottleneck.** `mem_read → mask_decode` is now `59.05 ms/frame`, still about 62% of total time.
2. **The biggest remaining hotspot is not TensorRT.** `affinity_topk` alone costs `21.50 ms/frame`; the current implementation is streamed/chunked PyTorch and should be replaced by a fused Triton/CUDA kernel.
3. **`sparse_readout` is now large enough to justify dedicated optimization.** At `15.23 ms/frame`, gather/reduce over top-k memory values is a first-class bottleneck.
4. **The object transformer is no longer the first thing to attack.** The SDPA/additive-mask cleanup brought it to `13.08 ms/frame`; it is still meaningful, but behind affinity and sparse readout.
5. **`aux_mask` is solved.** At `0.39 ms/frame`, mask construction is no longer worth targeting.
6. **Further TensorRT work is lower priority than readout kernels.** Encoder TRT and mask-decoder TRT are already doing the heavy lifting; the next meaningful gain is in memory readout.

## What would help next

1. **Fused affinity kernel**: replace the current chunked PyTorch `affinity_topk` path with a Triton/CUDA kernel that computes similarity, keeps running top-k, and softmax-normalizes in one pass.
2. **Fused sparse readout kernel**: replace gather + multiply + sum with a kernel that consumes top-k indices/weights directly. Longer-term, consider storing memory values in a token-major layout optimized for this path.
3. **Config tuning**: `top_k` (30→16), `max_num_tokens` (10000→5000), `mem_every` (5→8 or 10) — trades quality for speed on the remaining 59ms readout stage.
4. **TRT for encode_mask** (12ms): still useful, but lower priority than fused readout work.
5. **Future TRT versions**: retry readout engine — the wrapper code is ready, but TRT 10.15 still produces incorrect output.
6. **Lower resolution**: `--internal-size 720` remains the simplest way to move toward ~20 FPS.

## Scaling behavior

| Internal Size | Baseline | With TRT | FPS |
|---------------|---------|----------|-----|
| 480 | 41.7 ms | ~25 ms | ~40 |
| 720 | 74.7 ms | ~50 ms (est) | ~20 |
| 1080 | 162.8 ms | 95.8 ms | 10.4 |
