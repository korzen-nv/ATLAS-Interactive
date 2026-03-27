# ATLAS-Interactive: Optimization Status

Branch: `feature/blackwell` | Target: NVIDIA Blackwell 96GB

---

## Applied Optimizations

### 1. Config tuning for 96GB VRAM (committed: `236362f`)

**Files:** `gui/cutie/config/gui_config.yaml`, `gui/gui.py`

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `max_internal_size` | 480 | 1080 | Biggest quality win; ~5x more tokens per frame |
| `flip_aug` | False | True | Averages original + flipped predictions; ~2x VRAM |
| `mem_every` | 5 | 3 | More frequent memory updates |
| `stagger_updates` | 5 | 3 | Match mem_every |
| `top_k` | 30 | 50 | Broader attention over memory tokens |
| `max_mem_frames` | 10 | 15 | Keep 15 recent frames in working memory |
| `min_mem_frames` | 5 | 8 | Retain 8 when compressing to long-term |
| `num_prototypes` | 128 | 256 | Richer compressed representations |
| `max_num_tokens` | 10000 | 30000 | 3x more long-term memory capacity |
| `buffer_tokens` | 2000 | 5000 | Larger buffer before pruning |

GUI spinbox limits raised (work_mem max 100->500, long_mem max 100K->1M) to allow runtime tuning on high-VRAM cards.

### 2. top_k safety clamp (committed: `236362f`)

**File:** `gui/cutie/model/utils/memory_utils.py`

Added `top_k = min(top_k, similarity.shape[1])` before `torch.topk()` to prevent crash when memory has fewer tokens than top_k (first few frames of propagation).

### 3. CUDA inference optimizations (uncommitted)

**File:** `gui/main_controller.py` (`initialize_networks`)

```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
```

- **cuDNN benchmark**: Auto-tunes convolution algorithms for the input size on first run. Since propagation uses the same resolution every frame, the tuning pays off immediately. Typical gain: 5-15%.
- **TF32 matmul precision**: Enables TF32 on Ampere/Blackwell tensor cores. Uses 19-bit mantissa instead of 23-bit for FP32 matmuls -- nearly 2x throughput for matrix ops with negligible precision loss. Directly speeds up attention layers and linear projections in CUTIE's object transformer.

### 4. channels_last memory format (uncommitted)

**Files:** `gui/main_controller.py`, `gui/cutie/inference/inference_core.py`

Applied `torch.channels_last` (NHWC) memory format to:
- The CUTIE model weights (at load time)
- Input image tensors (after unsqueeze to 4D in inference_core)

cuDNN and tensor cores on Ampere+ natively operate in NHWC, so this eliminates per-layer format conversion overhead in all convolution operations.

### 5. torch.compile for pixel encoder (uncommitted)

**File:** `gui/main_controller.py`

```python
self.cutie.pixel_encoder = torch.compile(self.cutie.pixel_encoder)
self.cutie.key_proj = torch.compile(self.cutie.key_proj)
```

Compiles the ResNet backbone and key projection layers with Inductor. These are the most expensive per-frame components and have fixed input shapes during propagation, making them ideal compile targets. First frame has ~5-10s warmup, then every subsequent frame benefits from fused kernels.

**Note:** `mode='reduce-overhead'` (CUDA graphs) was tried but fails because `ImageFeatureStore` caches output tensors across frames, and CUDA graph replays overwrite those buffers. Default mode uses Inductor codegen without CUDA graphs.

**Cannot compile:** memory manager (dynamic memory size), object transformer (dynamic num_objects), mask encoder/decoder (chunking loops + `autocast(enabled=False)` blocks), `read_memory` (top-k with dynamic k).

### 6. Fast propagation mode (uncommitted)

**Files:** `gui/main_controller.py`, `gui/gui.py`

New "Fast forward" / "Fast backward" buttons (Shift+F / Shift+B) that propagate without per-frame UI rendering:

**Skipped per frame:**
- `show_current_frame()` -- no GPU overlay compositing, no visualization `.cpu()` transfer, no QImage/QPixmap
- `update_memory_gauges()` -- no gauge widget updates
- `save_visualization()` / `save_soft_mask()` -- no disk writes during loop
- `process_events()` only every 20 frames (just enough for pause to work)

**Still runs per frame:**
- `processor.step()` -- actual model inference (unchanged)
- `torch_prob_to_numpy_mask()` -- argmax + `.cpu()` (~2ms, needed for saving)
- Masks collected in a CPU list

**On finish:** renders the final frame, drains all deferred masks to disk via a background thread so the UI returns immediately.

**Measured result:** negligible speed difference vs normal propagation (~25 fps both). This confirms the bottleneck is `processor.step()`, not UI overhead. The visualization was already overlapped with data loading. The fast mode is still useful for reduced UI distraction during long runs.

### 7. FPS measurement label (uncommitted)

**Files:** `gui/main_controller.py`, `gui/gui.py`

Both normal and fast propagation now measure wall-clock time and display FPS after completion in a label next to the progress bar. Format: `Propagate: 25.3 fps (500 frames)` / `Fast: 25.8 fps (500 frames)`.

---

## Remaining Optimizations (not yet applied)

### High impact, low risk

**1. Remove hot-path `torch.cuda.empty_cache()`**
- Location: `gui/ritm/controller.py:42`
- The click path flushes the CUDA allocator on every interaction. This forces a full sync + allocator defrag. On a 96GB card it's pointless.
- Keep cache clearing only in explicit maintenance actions (`on_clear_memory`, `on_clear_non_permanent_memory`).

**2. BF16 autocast instead of FP16**
- Location: `gui/main_controller.py` (all `autocast()` calls)
- Blackwell natively supports BF16 with higher throughput than FP16. Change `autocast(self.device, ...)` to `autocast(self.device, dtype=torch.bfloat16, ...)`.
- Requires testing: CUTIE has 6 `autocast(enabled=False)` blocks that force FP32 for numerical stability. Each should be validated with BF16 before removal.

**3. Propagation data loader tuning**
- Location: `gui/reader.py`
- Enable `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2` on the DataLoader.
- Won't help much at 25fps (GPU-bound), but becomes relevant if model inference speeds up.

### Medium impact, medium risk

**4. Move RITM interactive mode to GPU distance maps**
- Location: `gui/click_controller.py:8` (`cpu_dist_maps=True`)
- The f-BRS optimizer runs SciPy L-BFGS through NumPy, shuttling data CPU<->GPU.
- Add a `high_vram` preset: `cpu_dist_maps=False`, optional BRS bypass for low-latency clicks.
- Risk: changes click segmentation quality slightly.

**5. Increase RITM click resolution**
- Location: `gui/click_controller.py:7` (`max_size=800`)
- Raise to 1400-1600 for Blackwell. Currently the click refinement runs at lower resolution than the propagation.

**6. Reduce GPU-to-CPU sync during propagation**
- Location: `gui/interactive_utils.py:171`, `gui/main_controller.py:333`
- Decouple render cadence from propagation cadence: render every Nth frame during propagation instead of every frame.
- Skip `save_soft_mask` and `save_visualization` by default during propagation runs.

**7. Fix GPU memory gauge**
- Location: `gui/main_controller.py:581`
- The torch memory gauge divides by 1024 incorrectly, making the displayed percentage misleading on a 96GB card.
- Change: `round(used_by_torch / global_total * 100 / 1024)` -> `round(used_by_torch / global_total * 100)`

### Low impact or high risk

**8. torch.compile on more components**
- `mask_encoder`, `mask_decoder`, `pixel_fuser` could be compiled if chunking is disabled (`chunk_size: -1`, already the case).
- Blocked by: `autocast(enabled=False)` blocks cause graph breaks. Need to audit each one for BF16 safety and remove where possible.
- The object transformer and memory manager cannot be compiled (dynamic shapes, `torch.where` indexing).

**9. Offline batched propagation mode**
- Process multiple future frames in parallel instead of one-by-one.
- Would require fundamental changes to the memory manager (it assumes sequential frame processing).
- This is the only way to truly saturate a 96GB GPU, but it's a large project.

**10. CUDA streams for async H2D transfer**
- Overlap CPU->GPU image copy with the previous frame's model inference.
- Marginal gain at current throughput (GPU-bound), but compounds with other speedups.

**11. torch.compile the RITM model**
- RITM (click segmentation) is a separate model that could benefit from compilation.
- Lower priority since clicks are infrequent compared to propagation.

---

## What we tried and rejected

| Approach | Why rejected |
|----------|-------------|
| `torch.compile(mode='reduce-overhead')` | CUDA graphs overwrite cached tensors in `ImageFeatureStore`. Would need to clone outputs after every encoder call, negating the speedup. |
| `max_mem_frames: 50` + `max_num_tokens: 100000` | OOM after ~30s of propagation at 1080px + flip_aug |
| Fast propagation (skip UI) for speed | Measured ~25fps both modes -- model inference dominates, not UI overhead. Kept for UX (less visual noise during long runs). |
