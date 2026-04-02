# Repository Guidelines

## Project Structure & Module Organization

`gui.py` is the main launcher for the interactive app. Core code lives under `gui/`:
- `gui/cutie/`: CUTIE-based propagation, memory, transformer, and config code.
- `gui/ritm/`: interactive click segmentation components.
- `gui/backends/`: model/backend adapters used by the GUI.
- `gui/trt_engine.py`, `gui/torch_rt.py`: TensorRT and runtime integration.

Tests live in `tests/` and currently focus on core numerical paths such as memory utils, transformer layers, and positional encoding. Example inputs are in `examples/`; figures and docs live in `figures/`, `README.md`, `OPTIMS.md`, and `SAM.md`.

## Build, Test, and Development Commands

- `pip install -e .`
  Installs the project in editable mode.
- `uv run gui.py --video examples/example.mp4`
  Launches the GUI against the bundled example video.
- `uv run python -m unittest`
  Runs the full test suite.
- `uv run python -m unittest tests.test_memory_utils`
  Runs a focused regression test for a single subsystem.
- `uv run gui.py --video ../ATLAS-Interactive/data/cmr-hd/episode_000001.mp4 --internal-size 1080 --profile --auto-propagate-forward --auto-pause-after 20`
  Standard profiling run for CUTIE propagation.

If you are not using `uv`, activate your environment first and run the same `python ...` commands directly.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep lines near the configured 100-column limit. Formatting is based on YAPF settings in [`pyproject.toml`](/home/pkorzeniowsk/Projects/atlas/ATLAS-Interactive-CUDA/pyproject.toml). Follow existing naming:
- `snake_case` for functions, variables, and files
- `CamelCase` for classes
- descriptive config keys such as `top_k`, `readout_backend`, `internal_size`

Prefer small, targeted changes. Preserve existing tensor shape comments and profiling labels when editing hot paths.

## Testing Guidelines

Use `unittest` and place new coverage in `tests/test_*.py`. Add focused regression tests for:
- numerical equivalence against reference implementations
- CUDA/Triton fast paths when relevant
- config-sensitive behavior and edge shapes

For performance work, reset `workspace/episode_000001.mp4/masks/` before each run so only `0000000.png` remains, then capture the `--profile` output and report before/after numbers in the PR.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase, feature-focused subjects such as `triton/cuda readout` and `tensorrt v3`. Keep commit titles concise and scoped to one change.

PRs should include:
- a brief summary of behavior changes
- linked issue or context
- test coverage added or run
- profiler deltas for optimization work
- screenshots or GIFs for GUI-visible changes
