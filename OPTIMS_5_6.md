# Optimization Status — torch.compile & fast propagation

## This commit

### 5. torch.compile for pixel encoder

**File:** `gui/main_controller.py` (`initialize_networks`)

```python
self.cutie.pixel_encoder = torch.compile(self.cutie.pixel_encoder)
self.cutie.key_proj = torch.compile(self.cutie.key_proj)
```

Compiles the ResNet backbone and key projection layers with Inductor. These are the most expensive per-frame components and have fixed input shapes during propagation, making them ideal compile targets. First frame has ~5-10s warmup, then every subsequent frame benefits from fused kernels. Wrapped in try/except for graceful fallback.

**`mode='reduce-overhead'` was tried and rejected** — it uses CUDA graphs which reuse fixed memory addresses. `ImageFeatureStore` caches encoder output tensors across frames, and graph replays overwrite those buffers. Default mode uses Inductor codegen without CUDA graphs.

**Cannot compile:** memory manager (dynamic memory size T), object transformer (dynamic num_objects), mask encoder/decoder (chunking loops + `autocast(enabled=False)` blocks), `read_memory` (top-k with dynamic k).

### 6. Fast propagation mode

**Files:** `gui/main_controller.py`, `gui/gui.py`

New "Fast forward" / "Fast backward" buttons (keyboard: Shift+F / Shift+B) that propagate without per-frame UI rendering.

**Skipped per frame:**
- `show_current_frame()` — no GPU overlay compositing, no visualization `.cpu()` transfer, no QImage/QPixmap creation
- `update_memory_gauges()` — no gauge widget updates
- `save_visualization()` / `save_soft_mask()` — no disk writes during loop
- `process_events()` only every 20 frames instead of every frame (just enough for pause to work)

**Still runs per frame:**
- `processor.step()` — actual model inference (unchanged)
- `torch_prob_to_numpy_mask()` — argmax + `.cpu()` (~2ms, needed for saving)
- Masks collected in a CPU list, not written to disk during propagation

**On finish/pause:**
- Renders the final frame normally with full visualization
- Drains all deferred masks to disk via a background thread so the UI returns immediately
- Displays FPS in the label (e.g. `Fast: 25.8 fps (500 frames)`)

**GUI changes:**
- Two new buttons in control bar: "Fast forward", "Fast backward"
- Keyboard shortcuts: Shift+F, Shift+B
- All propagation buttons (normal + fast) are mutually exclusive during propagation
- Both fast buttons become "Pause fast" when active

**Measured result:** Negligible speed difference vs normal propagation (~25 fps both modes at 1080px). This confirms the bottleneck is `processor.step()`, not UI overhead — the visualization was already overlapped with data loading. The fast mode is still useful for reduced UI distraction during long annotation runs.
