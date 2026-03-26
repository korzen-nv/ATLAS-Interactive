# CODEX Optimizations for RTX Blackwell 96GB

Date: 2026-03-25

This note captures a static review of the current codebase with a focus on making better use of a 96GB NVIDIA Blackwell-class GPU. This was not benchmarked locally because the workspace does not currently have a runnable PyTorch environment.

## Executive Summary

The main issue is not that the code lacks Blackwell-specific kernels. The bigger problem is that the app still presents a small, latency-oriented workload to the GPU:

- CUTIE is hard-capped to small internal resolutions and a short memory horizon.
- RITM interactive refinement is partly CPU-bound.
- Propagation is fed serially from CPU without much overlap.
- The GUI frequently forces GPU-to-CPU sync for rendering and saving.

A 96GB GPU will help immediately if the app spends that memory on:

1. higher internal resolution,
2. much larger long-term memory,
3. larger click/refinement resolution,
4. better I/O overlap,
5. an offline batched mode for propagation/refinement.

## Highest-Priority Findings

### 1. Remove hot-path CUDA cache clears

The click path flushes the CUDA allocator on every interaction:

- `gui/ritm/controller.py:42`

This is likely adding avoidable synchronization and allocator churn. Cache clearing should stay as an explicit UI action, not part of the normal click loop.

## Recommendation

- Remove `torch.cuda.empty_cache()` from the per-click path.
- Keep cache clearing only in explicit maintenance actions like:
  - `gui/main_controller.py:647`
  - `gui/main_controller.py:657`

### 2. Stop forcing RITM interactive mode onto the CPU-heavy path

The interactive click controller loads RITM with CPU distance maps:

- `gui/click_controller.py:8`

It also defaults to `f-BRS-B`:

- `gui/click_controller.py:17`

That path runs SciPy L-BFGS and shuttles data through NumPy and Torch:

- `gui/ritm/inference/predictors/brs.py:110`
- `gui/ritm/inference/predictors/brs.py:121`
- `gui/ritm/inference/predictors/brs_functors.py:41`

This is fine for compatibility, but it will not exploit a 96GB GPU well.

## Recommendation

- Add a `blackwell` or `high_vram` interactive preset.
- Set `cpu_dist_maps=False`.
- Make BRS optional for interactive mode.
- Prefer a pure forward-pass mode for low-latency clicks.
- Keep BRS as an optional offline refinement mode.

### 3. Raise the artificial workload caps

Current defaults are conservative:

- `gui/cutie/config/gui_config.yaml:17` sets `max_internal_size: 480`
- `gui/cutie/config/gui_config.yaml:22` sets `max_overall_size: 1080`
- `gui/cutie/config/gui_config.yaml:48` sets `long_term.max_mem_frames: 10`
- `gui/cutie/config/gui_config.yaml:51` sets `long_term.max_num_tokens: 10000`
- `gui/click_controller.py:7` caps RITM at `max_size=800`

Those values are the clearest reason VRAM will sit idle.

## Recommendation

Create a Blackwell preset roughly like:

```yaml
amp: True
max_internal_size: 960        # or 1080 after profiling
max_overall_size: -1          # keep native resolution when possible
mem_every: 4
top_k: 50
chunk_size: -1

long_term:
  count_usage: True
  max_mem_frames: 32
  min_mem_frames: 12
  num_prototypes: 256
  max_num_tokens: 262144
  buffer_tokens: 32768
```

And in the click controller:

- increase `max_size` from `800` to `1400` or `1600`,
- increase the zoom-in target from `480` to something closer to the actual display/input resolution.

Notes:

- `262144` long-term tokens is a reasonable first pass for 1080p-scale inference.
- This should be profiled before pushing further.

### 4. Propagation is CPU-fed and mostly serial

Propagation currently reads one frame at a time, tensorizes it on CPU, and then copies it to CUDA:

- `gui/reader.py:39`
- `gui/reader.py:40`
- `gui/main_controller.py:422`
- `gui/main_controller.py:430`

The loader also misses standard throughput features:

- `gui/reader.py:48`

It uses `batch_size=None` and does not enable:

- `pin_memory`
- `persistent_workers`
- `prefetch_factor`

## Recommendation

- Enable pinned memory on CUDA.
- Use persistent workers on Linux.
- Add a `prefetch_factor`.
- Introduce a dedicated CUDA stream for H2D copies during propagation.
- Consider pre-decoding or RAM-caching frame windows for long runs.

### 5. The runtime path is still generic CUDA, not tuned inference

The GUI wraps inference in autocast:

- `gui/main_controller.py:274`
- `gui/main_controller.py:402`
- `gui/main_controller.py:463`

That is useful, but the current codebase is still missing several obvious inference optimizations:

- no explicit BF16 policy,
- no `torch.compile`,
- no `channels_last`,
- no cuDNN benchmarking/tuning setup,
- no clear TF32 policy.

CUTIE also forces several internal sections back to FP32:

- `gui/cutie/model/modules.py:62`
- `gui/cutie/model/modules.py:79`
- `gui/cutie/model/big_modules.py:289`
- `gui/cutie/model/cutie.py:119`
- `gui/cutie/model/transformer/object_summarizer.py:78`
- `gui/cutie/utils/tensor_utils.py:48`

Some of those FP32 blocks may be necessary for stability, but they should be reviewed rather than assumed.

## Recommendation

- Move device handling to `torch.device` consistently.
- Gate CUDA logic on `device.type == "cuda"` instead of string equality.
- Prefer BF16 autocast on Blackwell-class hardware.
- Try `torch.compile` on CUTIE first.
- Use `channels_last` for image tensors and convolution-heavy models where safe.
- Enable:

```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
```

- Review each `autocast(enabled=False)` block and keep only the ones that are numerically necessary.

### 6. The app keeps forcing GPU-to-CPU sync for visualization and saving

The render/save path converts tensors back to CPU frequently:

- `gui/interactive_utils.py:18`
- `gui/interactive_utils.py:171`
- `gui/main_controller.py:333`

This is reasonable for a GUI, but it limits throughput during propagation.

## Recommendation

- Decouple render cadence from propagation cadence.
- Do not save soft masks by default during long propagation runs.
- Only materialize CPU images/masks when:
  - the frame is displayed,
  - the frame is explicitly saved,
  - the user pauses or commits.

### 7. The current interactive design cannot saturate a 96GB GPU

Interactive refinement is fundamentally batch-1 in the current implementation:

- `gui/interaction.py:77`
- `gui/interaction.py:83`
- `gui/ritm/inference/predictors/base.py:64`

That means Blackwell mostly buys:

- higher native resolution,
- larger memory windows,
- lower latency,
- more headroom for multiple objects.

It does not guarantee high utilization unless the app is changed to present a larger workload.

## Recommendation

Add an offline or batched mode that can:

- propagate multiple future frames together,
- refine multiple objects together,
- process a queue of click jobs,
- batch export/render operations.

Without that, the GPU will remain underfilled during normal GUI use.

## Secondary Findings

### Debug sync in the BRS hot path

There is a debug print in the predictor:

- `gui/ritm/inference/predictors/brs.py:72`

This should be removed from any performance-sensitive path.

### Incorrect GPU memory gauge

The Torch memory gauge appears to divide by `1024` again:

- `gui/main_controller.py:579`

That makes the displayed Torch-memory percentage misleading, which is especially unhelpful when tuning a 96GB card.

### Resource pipeline remains disk-centric

Frame extraction and loading still route through disk and PIL/OpenCV:

- `gui/resource_manager.py:178`
- `gui/resource_manager.py:244`

This is acceptable for compatibility, but not ideal for high-throughput propagation on a large GPU.

## Suggested Implementation Order

1. Remove hot-path `empty_cache()` and debug prints.
2. Add a Blackwell preset with larger CUTIE/RITM resolutions and a much larger long-term memory budget.
3. Switch interactive RITM to GPU distance maps and make BRS optional.
4. Upgrade the propagation loader with pinned memory, worker persistence, prefetching, and asynchronous H2D copies.
5. Add a modern inference runtime path: BF16, `channels_last`, `torch.compile`, cuDNN benchmark, TF32.
6. Reduce unnecessary GPU-to-CPU sync during rendering and saving.
7. Add an offline batched mode if the goal is to truly exploit 96GB of VRAM.

## Immediate Code Targets

If implementing this in code, start here:

- `gui/click_controller.py`
- `gui/ritm/controller.py`
- `gui/reader.py`
- `gui/main_controller.py`
- `gui/cutie/config/gui_config.yaml`
- `gui/cutie/inference/inference_core.py`
- `gui/cutie/inference/memory_manager.py`
- `gui/interactive_utils.py`

## Caveat

This document is based on static code inspection. The next step should be a short profiling pass on the target Blackwell machine to validate:

- stable BF16 behavior,
- best `max_internal_size`,
- long-term memory budget before quality returns diminish,
- whether `torch.compile` helps CUTIE without harming interactivity,
- whether the current RITM path should be split into:
  - low-latency interactive mode,
  - high-quality offline refinement mode.
