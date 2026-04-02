# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000001.mp4` (3600 frames, 1080p source)  
GPU: Blackwell (RTX PRO 6000), PyTorch 2.11 + CUDA 13.0, TensorRT 10.15, AMP FP16

## Current state: inference-core at ~69 ms/frame, propagation still limited by frame staging

### Inference-core profiler (steady-state average from a 25s run, first 50-frame block ignored)

Command used:

`uv run gui.py --video ../ATLAS-Interactive/data/cmr-hd/episode_000001.mp4 --internal-size 1080 --profile --auto-propagate-forward --auto-pause-after 25`

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.73 | 2.5% | **TRT engine** (FP16, static shapes) |
| Memory readout | affinity + sparse_readout + pixel_fusion + object_transformer | 32.14 | 46.5% | Triton sparse readout + Triton local-topk affinity + SDPA path |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 22.94 | 33.2% | **TRT engine** (FP16, static shapes, NO=19) |
| Add memory | encode_mask (ResNet18) + memory store | 12.21 | 17.7% | PyTorch |
| **Total** | | **69.17** | | **14.5 FPS** |

Steady-state blocks used for the averages above:

- Block 1: `TOTAL 69.14`, `affinity_topk 6.90`, `sparse_readout 1.49`
- Block 2: `TOTAL 69.11`, `affinity_topk 6.96`, `sparse_readout 1.52`
- Block 3: `TOTAL 69.01`, `affinity_topk 6.97`, `sparse_readout 1.46`
- Block 4: `TOTAL 69.43`, `affinity_topk 6.99`, `sparse_readout 1.53`

### Averaged step-profiler output

| Span | ms/frame |
|------|----------|
| `start → encode` | 0.15 |
| `encode → segment` | 1.73 |
| `segment → mem_read` | 0.00 |
| `mem_read → mask_decode` | 32.14 |
| `mask_decode → add_mem` | 22.94 |
| `add_mem → resize_up` | 12.21 |
| `resize_up → done` | 0.00 |
| `TOTAL (start → done)` | 69.17 |

### Detailed readout breakdown (steady-state, from `--profile`)

| Sub-stage | ms/frame | Notes |
|----------|----------|-------|
| `affinity_topk` | 6.96 | Major win from Triton local top-k using `tl.topk`; still not the final fully fused kernel |
| `sparse_readout` | 1.50 | Solved enough; token-major cache + Triton gather/reduction removed this as a first-class bottleneck |
| `object_transformer` | 13.02 | Still meaningful, but now clearly behind affinity |
| `pixel_fusion` | 5.38 | Secondary cost |
| `aux_mask` | 0.39 | Negligible |

### Propagation wall-clock profile (display enabled, same 25s run, first 50-frame block ignored)

This is the missing piece behind the user-visible propagation number. Even after inference-core dropped to `~69 ms/frame`, the GUI still reports about `~90 ms/frame` end to end with display enabled. The propagation wall profiler shows that the gap is not display; it is mostly frame staging into the model.

Steady-state propagation wall blocks used for the averages below:

- Block 1: `loader_wait 9.32`, `frame_upload 3.27`, `pending_total 3.74`
- Block 2: `loader_wait 9.71`, `frame_upload 3.31`, `pending_total 3.65`
- Block 3: `loader_wait 9.62`, `frame_upload 3.35`, `pending_total 3.73`
- Block 4: `loader_wait 9.95`, `frame_upload 3.38`, `pending_total 3.65`

| Propagation wall stage | ms/frame | Notes |
|------------------------|----------|-------|
| `loader_wait` | 9.65 | `DataLoader` wait, image fetch, and CPU `ToTensor()` work |
| `frame_upload` | 3.33 | CPU tensor to CUDA tensor upload |
| `pending_mask_to_numpy` | 2.61 | `torch.max(...).cpu().numpy()` for the previous frame |
| `pending_save_mask` | 0.23 | Queueing PNG mask writes is small in steady state |
| `pending_display_frame` | 0.10 | Display cost is tiny on average because the default throttle only shows every 16th frame |
| `pending_process_events` | 0.31 | Qt event processing is not a major cost |
| `pending_total` | 3.69 | All post-inference previous-frame handling combined |

Important note: with `--profile`, `InferenceCore.profiler.finish_frame()` calls `torch.cuda.synchronize()` every frame, so `step_dispatch` in the propagation wall profiler closely matches the CUDA step-profiler total. Even with that caveat, the non-inference gap is clear: `loader_wait + frame_upload` is about `13 ms/frame`, while display and save/UI work are negligible.

### Improvement over baseline at 1080p

| Stage | Baseline | Current | Speedup |
|-------|---------|---------|---------|
| Encoder | ~25 ms | 1.73 ms (TRT) | **14.5x** |
| Memory readout | ~71 ms | 32.14 ms | **2.21x** |
| Mask decoder | ~55 ms | 22.94 ms (TRT) | **2.40x** |
| Add memory | ~12 ms | ~12.21 ms | — |
| **Total** | **163 ms (6.1 FPS)** | **69.17 ms (14.5 FPS)** | **2.36x** |

### Improvement over previous optimized state

Previous state here means: TRT encoder + TRT mask decoder + Triton sparse readout + older affinity kernel path before the `tl.topk` local selector.

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| `affinity_topk` | 26.75 ms | 6.96 ms | **3.84x faster** |
| `sparse_readout` | 1.47 ms | 1.50 ms | Same |
| `object_transformer` | 13.02 ms | 13.02 ms | Same |
| `pixel_fusion` | 5.38 ms | 5.39 ms | Same |
| `aux_mask` | 0.38 ms | 0.39 ms | Same |
| `mem_read → mask_decode` | 51.89 ms | 32.14 ms | **1.61x faster** |
| **Total** | **88.71 ms (11.3 FPS)** | **69.17 ms (14.5 FPS)** | **1.28x faster** |

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

- **Sparse readout fast path**: `sparse_readout` now has a Triton gather/reduction kernel, and memory values are also cached in token-major layout at insert time. In steady-state profiling this dropped `sparse_readout` from `15.23 ms` to `1.47 ms`.
- **Affinity fast path**: `sparse_topk_affinity` now uses a real Triton kernel for the block-local similarity scan and local top-k on CUDA, then does a small global merge in PyTorch. Replacing the local repeated max/argmax loop with Triton `tl.topk` dropped `affinity_topk` from `26.75 ms` to `6.96 ms` in the step profiler.
- **Transformer fast path**: query transformer attention now uses PyTorch SDPA with a broadcast additive mask instead of per-head mask replication.
- **Cached positional encoding**: spatial PE is cached by shape/layout instead of batch/object count.
- **Propagation wall profiler**: `--profile` now also prints wall-clock propagation timings (`loader_wait`, `frame_upload`, `pending_mask_to_numpy`, `pending_display_frame`, etc.) so user-visible propagation FPS can be separated from model-only timings.

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

1. **The affinity optimization landed.** `affinity_topk` dropped from `26.75 ms/frame` to `6.96 ms/frame`, and total inference-core time dropped from `88.71 ms/frame` to `69.17 ms/frame`.
2. **The propagation gap is not display.** In steady-state wall profiling, `pending_display_frame` is only `0.10 ms/frame` average and `pending_process_events` is `0.31 ms/frame`. The OpenGL display path is not what is holding propagation at roughly `~90 ms/frame`.
3. **The main non-inference cost is frame staging.** `loader_wait` is `9.65 ms/frame` and `frame_upload` is `3.33 ms/frame`. Together they account for most of the remaining wall-time gap between model-only timings and user-visible propagation timings.
4. **The propagation post-processing path is relatively cheap.** `pending_total` is only `3.69 ms/frame`, with `pending_mask_to_numpy` at `2.61 ms/frame` and `pending_save_mask` at `0.23 ms/frame`.
5. **Within the model, memory readout is no longer the crisis it was.** `mem_read → mask_decode` is now `32.14 ms/frame` instead of `51.89 ms/frame`, but it is still the largest model stage.
6. **The next model-side bottlenecks changed.** `object_transformer` (`13.02 ms/frame`) and add-memory (`12.21 ms/frame`) are now in the same tier as the remaining affinity work.
7. **Encoder TRT and mask-decoder TRT remain solid wins.** There is no reason to revisit them before fixing the propagation data path.

## What would help next

1. **Fix the propagation input path before more model work**: the next wall-clock win is in `PropagationReader` / `DataLoader` / H2D staging, not the display path.
2. **Enable a faster host-to-device pipeline**: `pin_memory=True`, `persistent_workers=True`, and a pinned CPU tensor path would directly target the current `loader_wait + frame_upload` cost.
3. **Avoid per-frame `ToTensor()` when frames are already cached**: the workspace already preloads images into RAM; caching CPU tensors or CHW buffers would remove repeated tensorization work from the hot path.
4. **Make wall-clock benchmarks wait for preload completion**: when auto-propagation starts immediately, part of `loader_wait` may still be JPEG decode / cache-fill time. Benchmarking after preload is complete will show the true steady-state ceiling.
5. **After the propagation data path is tighter, revisit model bottlenecks**: the next inference-side targets are `object_transformer`, add-memory, and any remaining global-merge cost inside affinity.
6. **Config tuning remains available**: `top_k` (`30→16`), `max_num_tokens` (`10000→5000`), `mem_every` (`5→8` or `10`) still trade quality for speed.
7. **TRT for encode_mask**: still potentially useful, but no longer the most urgent problem.
8. **Future TRT versions**: retry the readout engine. The wrapper code is ready, but TRT 10.15 still produces incorrect output.
9. **Lower resolution**: `--internal-size 720` remains the simplest way to move toward ~20 FPS.

## Scaling behavior

| Internal Size | Baseline | Current | FPS |
|---------------|---------|---------|-----|
| 480 | 41.7 ms | ~25 ms | ~40 |
| 720 | 74.7 ms | ~50 ms (est) | ~20 |
| 1080 | 162.8 ms | 69.17 ms (model-only) | 14.5 |

Note: the current display-enabled propagation wall time at 1080p is still around `~90 ms/frame (~11 FPS)` because frame staging (`loader_wait + frame_upload`) has not been optimized yet.
