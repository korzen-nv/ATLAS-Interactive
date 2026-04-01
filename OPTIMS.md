# GPU Inference Optimization Benchmarks

Video: `cmr-hd/episode_000000.mp4` (3600 frames, 1080p source)
GPU: Blackwell, PyTorch 2.11 + CUDA 13.0, AMP FP16 always on

## Results

| Internal Size | Optimization | FPS | ms/frame | Speedup |
|---------------|-------------|-----|----------|---------|
| 480 (default) | None | 24.0 | 41.7 | baseline |
| 480 (default) | torch.compile | 22.6 | 44.3 | 0.94x (slower) |
| 720 | None | 13.4 | 74.7 | baseline |
| 1080 | None | 6.1 | 162.8 | baseline |
| 1080 | torch.compile | 6.3 | 159.8 | 1.02x |
| 1080 | FP8 weight-only | 6.3 | 159.3 | 1.02x |
| 1080 | torch.compile + FP8 | 6.3 | 158.0 | 1.03x |

## What was optimized

- `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)` on `pixel_encoder` (ResNet50) and `key_proj`
- FP8 weight-only quantization via torchao `Float8WeightOnlyConfig` on entire CUTIE model
- `mask_encoder` and `mask_decoder` could NOT be compiled (dynamic shapes in group ops / adaptive_avg_pool2d)

## Why gains are negligible

1. **Only ~30% of per-frame compute was optimized** (encoder + key_proj). The uncompiled memory readout (~30%) and mask decoder (~25%) dominate at high res.
2. **cuDNN ResNet convolutions are already near-optimal** -- Triton autotuned kernels were only ~10% faster than stock cuDNN on individual convolutions.
3. **FP8 weight-only saves little for conv layers** -- activation memory traffic (feature maps) dwarfs weight memory traffic; halving weight bandwidth gives minimal benefit.
4. **At 480, torch.compile is net negative** -- compilation overhead (graph tracing, recompilation on dynamic shapes) exceeds the small kernel-level gains.

## Scaling behavior (no optimizations)

| Internal Size | ms/frame | Relative to 480 |
|---------------|----------|-----------------|
| 480 | 41.7 | 1.0x |
| 720 | 74.7 | 1.8x |
| 1080 | 162.8 | 3.9x |

Roughly quadratic in spatial dimensions (1080/480 = 2.25x per side, 2.25^2 = 5x theoretical, actual 3.9x).

## What would actually help at 1080

- **TensorRT** for the encoder with static shapes (requires fixed resolution)
- **Optimizing memory readout**: reduce `top_k` (30->15), `max_num_tokens` (10000->5000)
- **Reducing `mem_every`**: fewer memory frames stored = smaller attention matrix
- **Profile-guided**: use `torch.profiler` to identify the actual hotspot at 1080 before optimizing further
