# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000001.mp4` (3600 frames, 1080p source)  
GPU: Blackwell (RTX PRO 6000), PyTorch 2.11 + CUDA 13.0, TensorRT 10.15, AMP FP16

## Current state: affinity fixed, propagation staging largely fixed

### Latest inference-core profiler (steady-state average from a 25s run, first 50-frame block ignored)

Command used:

`uv run gui.py --video ../ATLAS-Interactive/data/cmr-hd/episode_000001.mp4 --internal-size 1080 --profile --auto-propagate-forward --auto-pause-after 25`

| Stage | Component | ms/frame | % | Optimization |
|-------|-----------|----------|---|--------------|
| Encoder | pixel_encoder + pix_feat_proj + key_proj | 1.70 | 2.4% | **TRT engine** (FP16, static shapes) |
| Memory readout | affinity + sparse_readout + pixel_fusion + object_transformer | 33.56 | 46.8% | Triton sparse readout + Triton local-topk affinity + SDPA path |
| Mask decoder | decoder_feat_proc + upsample + predict + sensory GRU | 23.74 | 33.1% | **TRT engine** (FP16, static shapes, NO=19) |
| Add memory | encode_mask (ResNet18) + memory store | 12.67 | 17.7% | PyTorch |
| **Total** | | **71.75** | | **13.9 FPS** |

Steady-state blocks used for the averages above:

- Block 1: `TOTAL 71.47`, `affinity_topk 7.18`, `sparse_readout 1.55`
- Block 2: `TOTAL 71.10`, `affinity_topk 7.18`, `sparse_readout 1.55`
- Block 3: `TOTAL 71.72`, `affinity_topk 7.20`, `sparse_readout 1.63`
- Block 4: `TOTAL 71.96`, `affinity_topk 7.28`, `sparse_readout 1.60`
- Block 5: `TOTAL 72.52`, `affinity_topk 7.39`, `sparse_readout 1.56`

### Averaged step-profiler output

| Span | ms/frame |
|------|----------|
| `start → encode` | 0.07 |
| `encode → segment` | 1.70 |
| `segment → mem_read` | 0.00 |
| `mem_read → mask_decode` | 33.56 |
| `mask_decode → add_mem` | 23.74 |
| `add_mem → resize_up` | 12.67 |
| `resize_up → done` | 0.00 |
| `TOTAL (start → done)` | 71.75 |

### Detailed readout breakdown (steady-state, from `--profile`)

| Sub-stage | ms/frame | Notes |
|----------|----------|-------|
| `affinity_topk` | 7.25 | Major win from Triton local top-k using `tl.topk`; still not the final fully fused kernel |
| `sparse_readout` | 1.58 | Solved enough; token-major cache + Triton gather/reduction removed this as a first-class bottleneck |
| `object_transformer` | 13.60 | Still meaningful, but now clearly behind affinity |
| `pixel_fusion` | 5.64 | Secondary cost |
| `aux_mask` | 0.39 | Negligible |

### Propagation wall-clock profile after preload-aware direct prefetch (display enabled, same 25s run, first 50-frame block ignored)

This is the missing piece behind the user-visible propagation number. The latest run waits for preload completion before auto-start, bypasses multiprocess `DataLoader` when frames are already cached in RAM, stages a small pinned host window, and moves permutation/normalization onto the GPU. The propagation wall profiler shows that display is still not the issue, and that frame staging is much smaller than before.

Steady-state propagation wall blocks used for the averages below:

- Block 1: `loader_wait 3.71`, `frame_upload 0.02`, `pending_total 3.76`
- Block 2: `loader_wait 3.90`, `frame_upload 0.02`, `pending_total 3.67`
- Block 3: `loader_wait 3.85`, `frame_upload 0.02`, `pending_total 3.69`
- Block 4: `loader_wait 3.68`, `frame_upload 0.02`, `pending_total 3.54`
- Block 5: `loader_wait 3.78`, `frame_upload 0.02`, `pending_total 3.71`

| Propagation wall stage | ms/frame | Notes |
|------------------------|----------|-------|
| `loader_wait` | 3.78 | Main-process RAM-cache fetch plus prefetch bookkeeping |
| `frame_upload` | 0.02 | H2D upload moved almost entirely off the hot path via pinned-memory prefetch |
| `pending_mask_to_numpy` | 2.67 | `torch.max(...).cpu().numpy()` for the previous frame |
| `pending_save_mask` | 0.23 | Queueing PNG mask writes is still small in steady state |
| `pending_display_frame` | 0.10 | Display cost remains tiny on average because the default throttle only shows every 16th frame |
| `pending_process_events` | 0.26 | Qt event processing is not a major cost |
| `pending_total` | 3.67 | All post-inference previous-frame handling combined |

Important note: with `--profile`, `InferenceCore.profiler.finish_frame()` calls `torch.cuda.synchronize()` every frame, so `step_dispatch` in the propagation wall profiler closely matches the CUDA step-profiler total. Even with that caveat, the relative gain is clear: `loader_wait + frame_upload` dropped from about `13.0 ms/frame` to about `3.8 ms/frame`, while display and save/UI work remain negligible.

### Latest-only preview validation

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

This means the old no-skip penalty has been reduced, but not eliminated. The next propagation-side target is still `pending_mask_to_numpy`, followed by further simplification of no-skip UI updates if that mode matters.

### Improvement over baseline at 1080p

| Stage | Baseline | Current | Speedup |
|-------|---------|---------|---------|
| Encoder | ~25 ms | 1.73 ms (TRT) | **14.5x** |
| Memory readout | ~71 ms | 33.56 ms | **2.12x** |
| Mask decoder | ~55 ms | 23.74 ms (TRT) | **2.32x** |
| Add memory | ~12 ms | ~12.67 ms | — |
| **Total** | **163 ms (6.1 FPS)** | **71.75 ms (13.9 FPS)** | **2.27x** |

### Improvement over previous optimized state

Previous state here means: TRT encoder + TRT mask decoder + Triton sparse readout + older affinity kernel path before the `tl.topk` local selector.

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| `affinity_topk` | 26.75 ms | 7.25 ms | **3.69x faster** |
| `sparse_readout` | 1.47 ms | 1.58 ms | Same |
| `object_transformer` | 13.02 ms | 13.60 ms | Same tier |
| `pixel_fusion` | 5.38 ms | 5.64 ms | Same tier |
| `aux_mask` | 0.38 ms | 0.39 ms | Same |
| `mem_read → mask_decode` | 51.89 ms | 33.56 ms | **1.55x faster** |
| **Total** | **88.71 ms (11.3 FPS)** | **71.75 ms (13.9 FPS)** | **1.24x faster** |

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

1. **The affinity optimization is still holding.** `affinity_topk` is now about `7.25 ms/frame`, down from `26.75 ms/frame` before the Triton `tl.topk` selector landed.
2. **The propagation staging optimization landed too.** `loader_wait + frame_upload` fell from about `13.0 ms/frame` to about `3.8 ms/frame` after waiting for preload completion, bypassing multiprocess `DataLoader` for preloaded frames, and using pinned-memory async H2D.
3. **Display is still not the problem.** `pending_display_frame` remains about `0.10 ms/frame`, and `pending_process_events` remains small.
4. **The current non-inference remainder is small and mostly CPU mask materialization.** `pending_mask_to_numpy` is now the largest propagation-side non-model item at about `2.67 ms/frame`.
5. **Within the model, readout is no longer catastrophic but still largest.** `mem_read → mask_decode` is about `33.56 ms/frame`.
6. **The next model-side tier is now clearer.** `object_transformer` (`13.60 ms/frame`) and add-memory (`12.67 ms/frame`) are the main remaining heavy stages after affinity.
7. **Encoder TRT and mask-decoder TRT remain solid wins.** No reason to revisit them before the next propagation or model bottleneck is chosen.

## What would help next

1. **If propagation wall time is still the priority, attack mask materialization next**: `pending_mask_to_numpy` (`torch.max(...).cpu().numpy()`) is now the biggest non-model propagation-side item.
2. **Consider a latest-only preview path or true headless propagation mode**: display is already cheap, but removing preview/state updates from the hot loop would simplify the propagation path further.
3. **After that, go back to model bottlenecks**: `object_transformer`, add-memory, and any remaining global-merge cost inside affinity are now the main candidates.
4. **Config tuning remains available**: `top_k` (`30→16`), `max_num_tokens` (`10000→5000`), `mem_every` (`5→8` or `10`) still trade quality for speed.
5. **TRT for encode_mask**: still potentially useful, but not the most urgent issue.
6. **Future TRT versions**: retry the readout engine. The wrapper code is ready, but TRT 10.15 still produces incorrect output.
7. **Lower resolution**: `--internal-size 720` remains the simplest way to move toward ~20 FPS.

## Scaling behavior

| Internal Size | Baseline | Current | FPS |
|---------------|---------|---------|-----|
| 480 | 41.7 ms | ~25 ms | ~40 |
| 720 | 74.7 ms | ~50 ms (est) | ~20 |
| 1080 | 162.8 ms | 71.75 ms (model-only) | 13.9 |

Note: the latest propagation-side profiling shows that frame staging is much smaller than before. The remaining gap over model-only timings is now mostly `pending_mask_to_numpy`, not display rendering.
