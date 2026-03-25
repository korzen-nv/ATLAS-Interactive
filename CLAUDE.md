# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATLAS-Interactive is an interactive video labeling tool for clip-level anatomical segmentation in minimally invasive surgery. It combines keyframe annotation with temporal mask propagation using two ML models: CUTIE (primary) and RITM (alternative). Package name is `SurgeNetSeg`.

## Build & Run

```bash
# Install (editable mode, uses hatchling build backend)
pip install -e .

# Run the GUI
python gui.py --video examples/example.mp4
python gui.py --images /path/to/images
python gui.py --workspace /path/to/workspace

# Model weights auto-download from HuggingFace on first run
```

**Package manager**: uv (lock file: `uv.lock`). Dependencies in `pyproject.toml`. Requires Python >= 3.8, PyTorch 1.12+.

**No test suite or CI/CD exists in this repository.**

## Architecture

```
gui.py (entry point) → Hydra config + PyTorch device setup (CUDA/MPS/CPU)
    └── MainController (gui/main_controller.py) — central orchestrator
         ├── GUI (gui/gui.py) — PySide6 + pyqtdarktheme interface
         ├── ResourceManager (gui/resource_manager.py) — workspace I/O, frame cache, multithreaded saving
         └── InferenceCore (gui/cutie/inference/inference_core.py) — model forward pass
              ├── CUTIE Model (gui/cutie/model/cutie.py) — ResNet-50 encoder + Object Transformer
              ├── MemoryManager (gui/cutie/inference/memory_manager.py) — long-term temporal memory
              └── ObjectManager (gui/cutie/inference/object_manager.py) — multi-object tracking
```

**Data flow**: User loads video → ResourceManager extracts frames → GUI accepts click/polygon annotations → InferenceCore runs CUTIE model → masks propagated forward/backward through video → results saved to `workspace/`.

## Key Modules

- **gui/cutie/**: CUTIE model, inference engine, config (Hydra/OmegaConf YAML files), and utilities
- **gui/ritm/**: Alternative RITM model (DeepLabV3/HRNet architectures) with BRS prediction pipeline
- **gui/cutie/utils/palette.py**: Defines all 46 surgical anatomical classes and their RGB color mappings — edit here to customize classes/colors
- **gui/cutie/config/gui_config.yaml**: Main runtime config (workspace path, model weight paths, processing sizes, buffer/memory settings, output FPS)
- **gui/cutie/config/model/base.yaml**: Model architecture hyperparameters (ResNet-50 pixel encoder, transformer blocks/heads)
- **gui/cutie/utils/download_models.py**: Auto-download logic for model weights from HuggingFace

## Licensing

Code is MIT licensed. Model weights are CC-BY-NC-SA (non-commercial use only). These are separate license files: `LICENSE.txt` and `LICENSE_MODELS.txt`.
