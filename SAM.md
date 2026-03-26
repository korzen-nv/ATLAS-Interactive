# SAM Integration — `feature/sam` Branch Summary

## Overview

This branch introduces a **pluggable backend architecture** to ATLAS-Interactive, replacing the hard-coded CUTIE inference pipeline with a factory-based system that supports **CUTIE**, **SAM 2**, and **SAM 3** as interchangeable segmentation backends.

## Architecture

### Backend Protocol System (`gui/backends/`)

A new `gui/backends/` package defines two abstract protocols and a factory:

- **`PropagationBackend`** (`base.py`) — protocol for video mask propagation engines. Requires `step()`, `clear_memory()`, `clear_non_permanent_memory()`, `clear_sensory_memory()`, `update_config()`, `delete_objects()`, `output_prob_to_mask()`, and `get_memory_status()`.
- **`ClickBackend`** (`base.py`) — protocol for single-frame interactive click segmentation. Requires `interact()`, `unanchor()`, and `undo()`.
- **`MemoryStatus`** (`base.py`) — backend-agnostic dataclass for reporting memory utilization to the UI gauges.
- **`create_backends()`** (`factory.py`) — factory function that reads `cfg.backend` and returns a `(PropagationBackend, ClickBackend)` tuple.

### Backend Implementations

| Backend | Propagation | Click | Files |
|---------|-------------|-------|-------|
| **CUTIE** | `CutieBackend` (wraps `InferenceCore`) | `RitmClickBackend` (wraps `ClickController`) | `cutie_backend.py` |
| **SAM 2** | `Sam2PropagationBackend` (wraps `SAM2VideoPredictor`) | `Sam2ClickBackend` (wraps `SAM2ImagePredictor`) | `sam2_backend.py` |
| **SAM 3** | `Sam3PropagationBackend` (extends SAM 2, adds text prompts) | Reuses `Sam2ClickBackend` | `sam3_backend.py` |

### SAM 2 Backend Details

- Uses `SAM2VideoPredictor` for propagation with lazy `init_state()` on first frame.
- Anchor frames registered via `add_new_mask()`; propagation frames consume a generator from `propagate_in_video()`.
- Tracks permanent vs non-permanent anchors to support the existing memory clear/reset workflow.
- Handles Hydra config conflicts by reinitializing Hydra's global state to point at SAM 2's config directory.
- Supports `offload_video_to_cpu` and `offload_state_to_cpu` for memory efficiency.

### SAM 3 Backend Details

- Extends `Sam2PropagationBackend` with `add_text_prompt()` for concept-level segmentation (e.g., "grasper").
- Backward-compatible with all SAM 2 point/box interactions.
- Reuses `Sam2ClickBackend` for click-based interaction.

## Config Changes (`gui/cutie/config/gui_config.yaml`)

- New `backend` key: `cutie | sam2 | sam3` (default: `sam2`).
- New SAM 2 settings: `sam2_weights`, `sam2_model_cfg`.
- New SAM 3 settings: `sam3_weights`, `sam3_model_cfg`, `sam3_bpe_path`.
- Added Hydra searchpath for `sam2` package configs.
- Pre-configured with MedSAM2 weights (`MedSAM2_latest.pt`, `sam2.1_hiera_t.yaml`).

## Controller Changes (`gui/main_controller.py`)

- `initialize_networks()` now delegates to `create_backends()` factory instead of directly instantiating CUTIE/RITM.
- Network initialization moved **after** `ResourceManager` setup so `image_dir` is available for SAM backends.
- `self.processor` is assigned from the factory's propagation backend.
- Memory gauge updates use the backend-agnostic `get_memory_status()` API.
- CUTIE-specific memory tuning spinboxes guarded with `hasattr(self.processor, 'memory')`.
- GPU gauge updates wrapped in try/except for `CudaError` resilience.
- Fixed a `/1024` scaling bug in the torch memory gauge.

## Other Changes

- **`gui/interaction.py`** — Type hints updated from `ClickController` to `ClickBackend`.
- **`gui/interactive_utils.py`** — Color map padded to 256 entries to prevent crashes from out-of-range class IDs in stale masks.
- **`gui/cutie/utils/palette.py`** — Default 3-class palette overrides commented out (uses the full 46-class surgical palette instead).
- **`pyproject.toml`** — `requires-python` bumped from `>=3.8` to `>=3.10`; `sam2` added as a core dependency; optional dependency groups added for `desktop`, `sam2`, and `sam3`.

## Model Configs (`sam2.1/`)

Includes Hydra YAML configs for SAM 2.1 model variants:
- `sam2.1_hiera_base_plus.yaml`
- `sam2.1_hiera_l.yaml`
