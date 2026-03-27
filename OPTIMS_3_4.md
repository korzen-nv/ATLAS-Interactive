# Optimization Status — feature/blackwell branch

## Previously committed

### 1. Config tuning for 96GB VRAM (`236362f`)

Raised internal resolution from 480 to 1080px, enabled flip_aug, increased memory capacity (mem_every 5->3, top_k 30->50, max_mem_frames 10->15, num_prototypes 128->256, max_num_tokens 10K->30K). GUI spinbox limits raised to allow runtime tuning on high-VRAM cards.

### 2. top_k safety clamp (`236362f`)

Added `top_k = min(top_k, similarity.shape[1])` in `memory_utils.py` to prevent crash when memory has fewer tokens than top_k during first frames.

---

## This commit

### 3. CUDA inference optimizations

**File:** `gui/main_controller.py` (`initialize_networks`)

- **`cudnn.benchmark = True`** — Auto-tunes convolution algorithms for the input size on first run. Since propagation uses the same resolution every frame, the tuning cost is paid once and every subsequent frame benefits. Typical gain: 5-15%.
- **`set_float32_matmul_precision('high')`** — Enables TF32 on Ampere/Blackwell tensor cores. Uses 19-bit mantissa instead of 23-bit for FP32 matmuls, giving nearly 2x throughput for matrix ops with negligible precision loss. Speeds up attention layers and linear projections in CUTIE's object transformer.

### 4. channels_last memory format

**Files:** `gui/main_controller.py`, `gui/cutie/inference/inference_core.py`

- Model weights converted to NHWC layout at load time (`model.to(memory_format=torch.channels_last)`)
- Input image tensors converted after unsqueeze to 4D in inference_core

cuDNN and tensor cores on Ampere+ natively operate in NHWC. Without channels_last, every convolution layer silently converts NCHW->NHWC before computing and NHWC->NCHW after. This eliminates that overhead.

### 5. Propagation FPS label

**Files:** `gui/main_controller.py`, `gui/gui.py`

Added a `QLabel` next to the progress bar that displays throughput after propagation finishes (e.g. `Propagate: 25.3 fps (500 frames)`). Uses `time.monotonic()` wall-clock timing around the propagation loop. Useful for measuring the impact of optimizations without external tooling.
