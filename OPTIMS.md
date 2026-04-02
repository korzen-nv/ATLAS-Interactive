# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000001.mp4` (3600 frames, 1080p source)  
GPU: Blackwell (RTX PRO 6000), PyTorch 2.11 + CUDA 13.0, TensorRT 10.15, AMP FP16

## Current state: TRT encoder + TRT mask decoder + Triton sparse readout + partial Triton affinity

### Latest profiler (50 frames, 1080p, 19 objects, `--internal-size 1080`)

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.72 | 1.9% | **TRT engine** (FP16, static shapes) |
| Memory readout | affinity + sparse_readout + pixel_fusion + object_transformer | 51.82 | 58.6% | Triton sparse readout + partial Triton affinity + SDPA path |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 22.59 | 25.6% | **TRT engine** (FP16, static shapes, NO=19) |
| Add memory | encode_mask (ResNet18) + memory store | 12.10 | 13.7% | PyTorch |
| **Total** | | **88.36** | | **11.3 FPS** |

### Raw step-profiler output

| Span | ms/frame |
|------|----------|
| `start → encode` | 0.15 |
| `encode → segment` | 1.72 |
| `segment → mem_read` | 0.00 |
| `mem_read → mask_decode` | 51.82 |
| `mask_decode → add_mem` | 22.59 |
| `add_mem → resize_up` | 12.10 |
| `resize_up → done` | 0.00 |
| `TOTAL (start → done)` | 88.36 |

### Detailed readout breakdown (from `--profile`)

| Sub-stage | ms/frame | Notes |
|----------|----------|-------|
| `affinity_topk` | 26.86 | Largest remaining hotspot; current Triton path is only block-local top-k + PyTorch global merge, not the final kernel |
| `sparse_readout` | 1.42 | Major win; token-major cache + Triton gather/reduction removed this as a first-class bottleneck |
| `object_transformer` | 12.94 | Still meaningful, but now clearly behind affinity |
| `pixel_fusion` | 5.36 | Secondary cost |
| `aux_mask` | 0.38 | Negligible |

### Improvement over baseline at 1080p

| Stage | Baseline | Current | Speedup |
|-------|---------|---------|---------|
| Encoder | ~25 ms | 1.72 ms (TRT) | **14.5x** |
| Memory readout | ~71 ms | 51.82 ms | **1.37x** |
| Mask decoder | ~55 ms | 22.59 ms (TRT) | **2.43x** |
| Add memory | ~12 ms | ~12.10 ms | — |
| **Total** | **163 ms (6.1 FPS)** | **88.36 ms (11.3 FPS)** | **1.84x** |

### Improvement over previous optimized state

Previous state here means: TRT encoder + TRT mask decoder + streamed sparse top-k + PyTorch sparse readout.

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| `affinity_topk` | 21.50 ms | 26.86 ms | Worse |
| `sparse_readout` | 15.23 ms | 1.42 ms | **10.7x faster** |
| `object_transformer` | 13.08 ms | 12.94 ms | Same |
| `pixel_fusion` | 5.38 ms | 5.36 ms | Same |
| `aux_mask` | 0.39 ms | 0.38 ms | Same |
| `mem_read → mask_decode` | 59.1 ms | 51.82 ms | **1.14x faster** |
| **Total** | **95.8 ms (10.4 FPS)** | **88.36 ms (11.3 FPS)** | **1.08x faster** |

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
- Uses avg_pool2d instead of `F.interpolate(mode='area')` for ONNX compatibility
- 8 GB TRT workspace required for 1080p x 19 objects

## Other optimizations

- **Sparse readout fast path**: `sparse_readout` now has a Triton gather/reduction kernel, and memory values are also cached in token-major layout at insert time. This dropped `sparse_readout` from `15.23 ms` to `1.42 ms`.
- **Affinity fast path (partial)**: `sparse_topk_affinity` now uses a real Triton kernel for the block-local similarity scan and local top-k on CUDA, then does a small global merge in PyTorch. This compiles and runs at `top_k=30`, but it is not yet the right final kernel and is now the dominant bottleneck.
- **Transformer fast path**: query transformer attention now uses PyTorch SDPA with a broadcast additive mask instead of per-head mask replication.
- **Cached positional encoding**: spatial PE is cached by shape/layout instead of batch/object count.
- **Step profiler**: `--profile` prints both coarse stage timings and detailed readout sub-stages (`affinity_topk`, `sparse_readout`, `pixel_fusion`, `object_transformer`, `aux_mask`).

## TRT readout (pixel_fuser + object_transformer) — investigated, not viable

An ONNX wrapper (`_ReadoutPipelineONNXWrapper`) was implemented with:
- Manual multi-head attention (explicit `F.linear` + matmul + softmax) replacing `nn.MultiheadAttention`
- Float additive attention masks (`masked_fill` with `-inf`) replacing boolean masks
- Precomputed positional encoding
- Static-shape `_aux_mask` replacing `torch.where`

**Result**: Wrapper verified correct in PyTorch (diff max=0.025 vs original). However, TRT engine consistently produces incorrect output (silent corruption, not NaN after FP16 fix). Root cause is in TRT 10.15's ONNX graph handling, not cache, precision, I/O binding, or wrapper logic. The wrapper code is kept in `trt_engine.py` for future TRT versions.

## Historical: previous optimized state before Triton sparse readout

This was the state with TRT encoder + TRT mask decoder + streamed sparse top-k + PyTorch sparse readout.

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.7 | 1.8% | **TRT engine** (FP16, static shapes) |
| Memory readout | affinity + sparse_readout + pixel_fusion + object_transformer | 59.1 | 61.7% | Streamed sparse top-k + SDPA path |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 22.9 | 23.9% | **TRT engine** (FP16, static shapes, NO=19) |
| Add memory | encode_mask (ResNet18) + memory store | 12.0 | 12.5% | PyTorch |
| **Total** | | **95.8** | | **10.4 FPS** |

Historical detailed readout breakdown:

| Sub-stage | ms/frame | Notes |
|----------|----------|-------|
| `affinity_topk` | 21.50 | Largest hotspot at the time; still chunked PyTorch, not a fused kernel |
| `sparse_readout` | 15.23 | Gather/reduction over top-k values was a first-class bottleneck |
| `object_transformer` | 13.08 | Improved with SDPA + lighter mask handling |
| `pixel_fusion` | 5.38 | Secondary cost |
| `aux_mask` | 0.39 | Negligible |

## Historical: torch.compile + FP8 benchmarks (pre-TRT)

| Internal Size | Optimization | FPS | ms/frame | Speedup |
|---------------|-------------|-----|----------|---------|
| 480 (default) | None | 24.0 | 41.7 | baseline |
| 480 (default) | torch.compile | 22.6 | 44.3 | 0.94x |
| 1080 | None | 6.1 | 162.8 | baseline |
| 1080 | torch.compile + FP8 | 6.3 | 158.0 | 1.03x |

## Findings

1. **The latest readout win is real.** `sparse_readout` fell from `15.23 ms/frame` to `1.42 ms/frame`, so the token-major cache plus the Triton gather/reduction path paid off.
2. **Affinity is now unequivocally the main bottleneck.** `affinity_topk` is `26.86 ms/frame`, larger than every other readout sub-stage by a wide margin.
3. **The current affinity fast path is not the final solution.** The present implementation is Triton for block-local similarity/top-k plus a PyTorch global merge; it is better than chunked PyTorch only in some synthetic cases and is worse in the full step profiler.
4. **Readout is still the dominant stage overall.** `mem_read → mask_decode` is `51.82 ms/frame`, about 59% of total frame time.
5. **The object transformer is no longer the first thing to attack.** At `12.94 ms/frame`, it matters, but it is clearly behind affinity.
6. **`add_mem → resize_up` is now a second-tier bottleneck.** At `12.10 ms/frame`, encode-mask/add-memory work is roughly tied with object-transformer cost.
7. **`aux_mask` is solved.** At `0.38 ms/frame`, mask construction is no longer worth targeting.
8. **Further TensorRT work is lower priority than affinity.** Encoder TRT and mask-decoder TRT are already doing the heavy lifting; the next meaningful gain is in memory readout, specifically affinity.

## What would help next

1. **Rewrite the affinity kernel**: replace the current block-local Triton + PyTorch merge path with a truly efficient Triton/CUDA kernel for similarity + top-k + softmax. The current kernel is not enough.
2. **Autotune affinity on Blackwell**: tune `BLOCK_N`, `BLOCK_HW`, `BLOCK_CK`, and `num_warps` for the actual deployment shape instead of using a single fixed configuration.
3. **Instrument affinity internally**: split timing inside `sparse_topk_affinity` into Triton similarity/local-topk, PyTorch global merge, and usage accumulation. That will show whether the remaining cost is in-kernel selection, the merge, or `usage.scatter_add_`.
4. **Attack object-transformer only after affinity**: `object_transformer` is now the next-largest readout component after affinity, but it is no longer the first thing to optimize.
5. **Attack add-memory only after affinity**: `encode_mask` / add-memory work is still around `12 ms/frame`; useful, but lower priority than affinity.
6. **Config tuning remains available**: `top_k` (`30→16`), `max_num_tokens` (`10000→5000`), `mem_every` (`5→8` or `10`) trade quality for speed on the remaining readout stage.
7. **TRT for encode_mask**: still potentially useful, but lower priority than affinity.
8. **Future TRT versions**: retry the readout engine. The wrapper code is ready, but TRT 10.15 still produces incorrect output.
9. **Lower resolution**: `--internal-size 720` remains the simplest way to move toward ~20 FPS.

## Scaling behavior

| Internal Size | Baseline | Current | FPS |
|---------------|---------|---------|-----|
| 480 | 41.7 ms | ~25 ms | ~40 |
| 720 | 74.7 ms | ~50 ms (est) | ~20 |
| 1080 | 162.8 ms | 88.36 ms | 11.3 |
