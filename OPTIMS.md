# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000001.mp4` (3600 frames, 1080p source)  
GPU: Blackwell (RTX PRO 6000), PyTorch 2.11 + CUDA 13.0, TensorRT 10.15, AMP FP16

## Current state: active-object reduction landed, propagation cleanup is now secondary

This workspace starts from only 7 non-empty object channels even though the GUI config allows 19 objects. CUTIE now tracks only the active objects internally, resolves an exact-object-count TRT mask decoder when possible, and expands the result back to canonical class channels for the GUI/controller. That kept the large speedup while fixing the earlier class-ID mix-up and restart crash.

Shorter burst runs previously reached about `25.0 ms/frame` steady-state at 1080p. The latest full 3599-frame validation is a little more conservative: about `28.5 ms/frame` steady-state at 1080p, and about `8.9 ms/frame` steady-state at 480p.

### Latest 1080p inference-core profiler (full-sequence steady-state block)

Command used:

`uv run gui.py --video ../ATLAS-Interactive/data/cmr-hd/episode_000001.mp4 --internal-size 1080 --profile`

Most recent steady-state block from the 3599-frame `--internal-size 1080` run:

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.80 | 6.3% | **TRT encoder** (FP16, static shapes) |
| Memory readout | affinity + sparse_readout + pixel_fusion + object_transformer | 14.73 | 51.6% | Active-object path + Triton sparse readout + Triton local-topk affinity + SDPA path |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 8.79 | 30.8% | **TRT decoder** on the reduced active-object path |
| Add memory | encode_mask (ResNet18) + memory store | 3.22 | 11.3% | Active-object path + channels-last / AMP cleanups |
| **Total** | | **28.54** | | **35.0 FPS model-only steady-state** |

Latest whole-run headline from the same profile pass:

- `Propagated 3599 frames in 129.2s (27.9 fps, 35.9 ms/frame)`

Important caveat: the whole-run number still includes startup, warmup, and the slow first block. The steady-state hot loop is materially faster than the headline FPS suggests.

### Averaged step-profiler output

| Span | ms/frame |
|------|----------|
| `start → encode` | 0.09 |
| `encode → segment` | 1.71 |
| `segment → mem_read` | 0.00 |
| `mem_read → mask_decode` | 14.73 |
| `mask_decode → add_mem` | 8.79 |
| `add_mem → resize_up` | 3.22 |
| `resize_up → done` | 0.00 |
| `TOTAL (start → done)` | 28.54 |

### Detailed readout breakdown (steady-state, from `--profile`)

| Sub-stage | ms/frame | Notes |
|----------|----------|-------|
| `affinity_query_precompute` | 0.02 | Negligible; the affinity cost is not in query-side prep |
| `affinity_triton_kernel` | 5.38 | Largest remaining affinity slice; this is the real readout hotspot inside the affinity path |
| `affinity_candidate_build` | 5.40 | Dominated by the Triton kernel above |
| `affinity_select_topk` | 2.09 | Second affinity slice; global select/merge over block candidates |
| `affinity_softmax` | 0.05 | Negligible |
| `affinity_usage` | 0.01 | Negligible |
| `sparse_readout` | 0.48 | No longer a serious bottleneck |
| `object_transformer` | 3.91 | Meaningfully smaller after the layout and active-object reductions |
| `pixel_fusion` | 1.65 | Secondary cost |
| `aux_mask` | 0.32 | Negligible |
| `trt_mask_prepare_inputs` | 0.13 | Small |
| `trt_mask_execute` | 8.42 | Dominates TRT decode; the decoder engine itself is still the next single heavy kernel |
| `trt_mask_decode` | 8.55 | Mostly engine execution, not Python-side setup |
| `mask_encoder` | 2.78 | Add-memory is now moderate rather than first-order |

### Latest 1080p propagation wall-clock profile (same recent steady-state block)

The propagation path is no longer the primary problem. Preload-aware staging and latest-only preview flushing already removed most of the old wall-time tax, and the recent GPU-side `uint8` narrowing of hard masks cut the old `pending_mask_to_numpy` cost by roughly 4x.

| Propagation wall stage | ms/frame | Notes |
|------------------------|----------|-------|
| `loader_wait` | 4.58 | Main-process RAM-cache fetch plus prefetch bookkeeping |
| `frame_upload` | 0.02 | H2D upload remains effectively off the hot path |
| `step_dispatch` | 28.44 | Closely tracks the CUDA step-profiler total when `--profile` is enabled |
| `pending_mask_to_numpy` | 0.54 | Much smaller after narrowing to `uint8` on GPU before D2H |
| `pending_save_mask` | 0.88 | Now the largest non-model steady-state item in the default path |
| `pending_display_frame` | 0.22 | Preview remains small with the default throttle |
| `pending_update_memory_gauges` | 0.50 | Small but visible now that mask materialization is cheaper |
| `pending_process_events` | 0.88 | Qt event processing is still secondary |
| `pending_total` | 3.02 | All post-inference previous-frame handling combined |

### Latest-only preview validation (historical no-skip check)

A second pass changed propagation preview from frame-count-driven paint calls to latest-only, time-based preview flushing. I also added `--display-skip` to make skip/no-skip runs reproducible from the command line.

No-skip validation command:

`uv run gui.py --video ../ATLAS-Interactive/data/cmr-hd/episode_000001.mp4 --internal-size 1080 --profile --auto-propagate-forward --auto-pause-after 15 --display-skip 0`

Steady-state no-skip wall profile:

| Propagation wall stage | ms/frame | Notes |
|------------------------|----------|-------|
| `loader_wait` | 3.61 | Still small after the staging optimization |
| `frame_upload` | 0.02 | Still effectively off the hot path |
| `pending_mask_to_numpy` | 2.71 | Largest non-model propagation-side item |
| `pending_display_frame` | 1.30 | Real no-skip preview paint cost |
| `pending_update_memory_gauges` | 0.88 | Gauge updates are now visible in no-skip mode |
| `pending_process_events` | 2.64 | Bounded event-pump cost for the live preview path |
| `pending_total` | 7.67 | No-skip preview/UI overhead is now bounded rather than per-frame paint driven |

This no-skip validation was captured before the later GPU-side `uint8` hard-mask narrowing, so the absolute `pending_mask_to_numpy` number is now lower in the default path. The main point still holds: no-skip preview has a real bounded UI cost, but the default throttled propagation path is now model-dominated.

### Improvement over baseline at 1080p

| Stage | Baseline | Current | Speedup |
|-------|---------|---------|---------|
| Encoder | ~25 ms | 1.80 ms (TRT) | **13.9x** |
| Memory readout | ~71 ms | 14.73 ms | **4.82x** |
| Mask decoder | ~55 ms | 8.79 ms (TRT active-object path) | **6.26x** |
| Add memory | ~12 ms | 3.22 ms | **3.73x** |
| **Total** | **163 ms (6.1 FPS)** | **28.54 ms (35.0 FPS model-only steady-state)** | **5.71x** |

### Improvement over the previous 71.75 ms state

Previous state here means the earlier optimized path documented in this file: TRT encoder + TRT decoder + Triton sparse readout + Triton local-topk affinity, but still tracking the full configured object set internally.

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| `mem_read → mask_decode` | 33.56 ms | 14.73 ms | **2.28x faster** |
| `mask_decode → add_mem` | 23.74 ms | 8.79 ms | **2.70x faster** |
| `add_mem → resize_up` | 12.67 ms | 3.22 ms | **3.93x faster** |
| `object_transformer` | 13.60 ms | 3.91 ms | **3.48x faster** |
| `pixel_fusion` | 5.64 ms | 1.65 ms | **3.42x faster** |
| `pending_mask_to_numpy` | 2.67 ms | 0.54 ms | **4.94x faster** |
| **`TOTAL (start → done)`** | **71.75 ms** | **28.54 ms** | **2.51x faster** |

### Latest scaling validation (full 3599-frame propagation)

These are the latest measured whole-run and steady-state numbers from the same video under the current active-object path.

| Internal Size | Whole-run FPS | Whole-run ms/frame | Steady-state step total | Steady-state step_dispatch | Main steady-state hotspots |
|---------------|---------------|--------------------|-------------------------|----------------------------|----------------------------|
| 480 | 66.9 | 15.0 | 8.88 | 9.20 | `object_transformer 2.55`, `trt_mask_execute 1.46`, `affinity_triton_kernel 0.66` |
| 1080 | 27.9 | 35.9 | 28.54 | 28.44 | `trt_mask_execute 8.42`, `affinity_triton_kernel 5.38`, `object_transformer 3.91` |

Observed scaling from 480 → 1080:

- `TOTAL (start → done)`: `8.88 ms` → `28.54 ms` (`3.21x`)
- `affinity_triton_kernel`: `0.66 ms` → `5.38 ms` (`8.15x`)
- `trt_mask_execute`: `1.46 ms` → `8.42 ms` (`5.77x`)
- `object_transformer`: `2.55 ms` → `3.91 ms` (`1.53x`)

This is useful because it shows the current 1080p penalty is driven much more by the affinity kernel and TRT decoder execution than by the object transformer.

## TensorRT engines

Two native TRT engines built at startup (ONNX export + TRT autotuner), cached to `~/.cache/atlas-trt/`. First build takes 10-30s; subsequent loads are instant. Cache key includes resolution, GPU, TRT version, FP16 flag, model signature, and weights mtime.

**Encoder engine** (`TRTEncoder`):
- Combines: PixelEncoder (ResNet50 layers 1-3) + pix_feat_proj (1x1 Conv) + KeyProjection (3 Conv2d + pow + sigmoid)
- Input: `(1, 3, H, W)` raw image (normalisation baked in)
- 7 outputs: f16, f8, f4, pix_feat, key, shrinkage, selection
- Conv-BN-ReLU fusion, FP16 kernels throughout

**Mask decoder engine** (`TRTMaskDecoderManager` + `TRTMaskDecoder`):
- Combines: DecoderFeatureProcessor + MaskUpsampleBlock (x2) + prediction Conv + SensoryUpdater GRU
- Manager lazily loads or builds exact-object-count decoder engines and keeps a larger fallback engine
- Inputs: f8_raw, f4_raw, memory_readout, sensory
- 2 outputs: new_sensory, logits
- Current workspace runs on the reduced active-object path (7 active objects in steady state)
- Falls back to zero-padding + output slicing only when no exact engine is available
- Uses avg_pool2d instead of `F.interpolate(mode='area')` for ONNX compatibility
- 8 GB TRT workspace required for 1080p x 19 objects

## Other optimizations

- **Sparse readout fast path**: `sparse_readout` now has a Triton gather/reduction kernel, and memory values are also cached in token-major layout at insert time. In steady-state profiling this dropped `sparse_readout` from `15.23 ms` to `1.47 ms`.
- **Affinity fast path**: `sparse_topk_affinity` now uses a real Triton kernel for the block-local similarity scan and local top-k on CUDA, then does a small global merge in PyTorch. Replacing the local repeated max/argmax loop with Triton `tl.topk` dropped `affinity_topk` from `26.75 ms` to `6.96 ms` in the step profiler.
- **Active-object inference path**: non-empty probability-mask channels are now inferred up front, CUTIE keeps only those objects internally, and the backend remaps sparse internal outputs back to canonical GUI class channels. This is the main reason the current workspace dropped from the earlier `71.75 ms` state to `24.98 ms` steady-state model time.
- **Dynamic TRT decoder selection**: the mask decoder path now resolves an exact-object-count TRT engine when possible instead of always paying for the configured 19-object engine.
- **Transformer fast path**: query transformer attention now uses PyTorch SDPA with a broadcast additive mask instead of per-head mask replication, the SDPA head split no longer forces an extra contiguous copy, and the PixelFFN path avoids an unnecessary channels-last round-trip.
- **Mask/add-memory cleanup**: `MaskEncoder` now runs channels-last, sensory updaters only force FP32 for the recurrent update, and exact-ratio downsampling takes an `avg_pool2d` fast path.
- **Propagation-side mask materialization**: hard masks are now narrowed to `uint8` on GPU before the D2H copy, cutting `pending_mask_to_numpy` from roughly `2.7-3.5 ms` to about `0.6-0.8 ms` in steady state.
- **Cached positional encoding**: spatial PE is cached by shape/layout instead of batch/object count.
- **Profiler detail**: `--profile` now also prints wall-clock propagation timings and finer model splits (`affinity_candidate_build`, `affinity_select_topk`, and later sub-slices for affinity / TRT decoder work) so the next bottleneck can be chosen from measurements instead of guesses.

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

1. **The active-object path is still the biggest recent win.** On this workspace, reducing the internal path from the configured 19 objects to the 7 actually active ones cut the 1080p steady-state model path from the older `71.75 ms` state to the latest `28.54 ms`.
2. **Propagation-side cleanup is now secondary.** At 1080p, `pending_mask_to_numpy` is only about `0.54 ms/frame`, and the whole pending block is about `3.02 ms/frame`.
3. **Readout is still the largest model stage.** `mem_read → mask_decode` is `14.73 ms/frame` at 1080p, and most of that is the affinity path rather than sparse readout or object-transformer overhead.
4. **The real affinity cost is the Triton kernel, not query prep.** `affinity_query_precompute` is about `0.02 ms/frame`, while `affinity_triton_kernel` is about `5.38 ms/frame`.
5. **TRT mask decode is still the next single heavy kernel.** `trt_mask_execute` is about `8.42 ms/frame`, and the rest of TRT decode setup is negligible.
6. **480p scales very well.** The same path runs at about `8.88 ms/frame` steady-state model time and `66.9 FPS` whole-run wall time at `--internal-size 480`.
7. **The remaining work should stay measurement-driven.** The next useful improvements are likely inside the affinity kernel or the TRT decoder engine, not broad architecture changes.

## What would help next

1. **Tune the Triton affinity kernel**: the new profiler slices show that the work is in `affinity_triton_kernel`, not `affinity_query_precompute`.
2. **Tune the TRT decoder engine**: the new profiler slices show that the work is in `trt_mask_execute`, not padding, allocation, or I/O binding.
3. **If user-visible propagation FPS matters after that, look at `loader_wait` and `pending_save_mask`**: those are now larger than `pending_mask_to_numpy` in the default path.
4. **Config tuning remains available**: `top_k` (`30→16`), `max_num_tokens` (`10000→5000`), `mem_every` (`5→8` or `10`) still trade quality for speed.
5. **TRT for encode_mask**: still potentially useful, but it is no longer the obvious next bottleneck.
6. **Future TRT versions**: retry the readout engine. The wrapper code is ready, but TRT 10.15 still produces incorrect output.
7. **Lower resolution**: `--internal-size 720` remains the simplest coarse knob if absolute FPS matters more than quality.

## Scaling behavior

| Internal Size | Baseline | Current | FPS |
|---------------|---------|---------|-----|
| 480 | 41.7 ms | 8.88 ms steady-state model / 15.0 ms whole-run wall | 66.9 whole-run |
| 720 | 74.7 ms | ~50 ms (est) | ~20 |
| 1080 | 162.8 ms | 28.54 ms steady-state model / 35.9 ms whole-run wall | 27.9 whole-run |

Note: the whole-run propagation numbers remain lower than the steady-state model FPS because startup, warmup, and the first slow block are included. In the default steady-state path, the remaining wall-time gap over model-only execution is now dominated by `loader_wait`, `pending_save_mask`, and other small CPU-side tasks rather than display rendering.
