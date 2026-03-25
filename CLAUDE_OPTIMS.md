# ATLAS-Interactive: Optimization & Research Notes

## GPU Optimization for 96GB Blackwell

### Config Tuning (`gui/cutie/config/gui_config.yaml`)

The default config targets consumer GPUs (~8GB VRAM). With 96GB, we can increase resolution and memory capacity significantly.

**Key constraint**: at 1080px + flip_aug, each frame produces ~2584 tokens (vs ~510 at 480px), and flip_aug doubles the batch dimension. So memory per frame is ~10x the original. You cannot simply multiply all parameters — frame counts must be scaled down relative to the resolution increase.

#### Recommended config for 96GB

```yaml
max_internal_size: 1080   # was 480 — biggest quality win
flip_aug: True             # was False — averages original + flipped predictions, ~2x VRAM cost

mem_every: 3               # was 5 — more frequent memory updates
stagger_updates: 3         # was 5 — match mem_every
top_k: 50                  # was 30 — broader attention over memory tokens

long_term:
  count_usage: True
  max_mem_frames: 15       # was 10 — keep 15 recent frames in working memory
  min_mem_frames: 8        # was 5 — retain 8 when compressing to long-term
  num_prototypes: 256      # was 128 — richer compressed representations
  max_num_tokens: 30000    # was 10000 — 3x more long-term capacity
  buffer_tokens: 5000      # was 2000
```

#### What we tried and failed (OOM at ~30s of propagation)

```yaml
# TOO AGGRESSIVE — exhausted 96GB in ~30s
max_mem_frames: 50         # 50 * 2584 * 2 (flip) = ~258K working tokens
min_mem_frames: 25
num_prototypes: 512
max_num_tokens: 100000
buffer_tokens: 10000
mem_every: 2
top_k: 100
```

#### VRAM budget breakdown (approximate)

| Component | 480px (original) | 1080px + flip_aug |
|---|---|---|
| Tokens per frame | ~510 (30x17) | ~2584 (68x38) |
| Batch multiplier | 1x | 2x (flip_aug) |
| Effective tokens/frame | ~510 | ~5168 |
| 15 working frames | 7,650 | 77,520 |
| Model weights (CUTIE + RITM) | ~200MB | ~200MB |
| Long-term memory (30K tokens) | N/A | ~2-4GB |
| Sensory memory (42 objects) | ~100MB | ~500MB |

#### Further tuning levers (if still OOM)

1. **Disable flip_aug** (saves ~2x VRAM) — set `flip_aug: False`
2. **Reduce resolution to 720** — set `max_internal_size: 720` (~60% of 1080px tokens)
3. **Reduce working frames** — set `max_mem_frames: 10` (back to default)
4. **Use chunk_size** — set `chunk_size: 10` to process objects in batches instead of all 42 at once

### Code changes for high-VRAM configs

1. **GUI spinbox limits raised** (`gui/gui.py`): work_mem max 100→500, long_mem max 100K→1M
2. **top_k safety clamp** (`gui/cutie/model/utils/memory_utils.py`): `top_k = min(top_k, similarity.shape[1])` prevents crash when memory has fewer tokens than top_k (e.g., first few frames)

---

## Memory Architecture (CUTIE)

CUTIE uses a three-tier memory system, which is its main architectural advantage over SAM-family models:

### Tier 1: Sensory Memory
- Per-object high-dimensional feature map (256-dim)
- Updated on a staggered schedule within each `mem_every` window
- Cleared at the start of each propagation (`clear_sensory_memory()`)

### Tier 2: Working Memory (KeyValueMemoryStore)
- Stores key/shrinkage/value tensors for recent frames
- Split into **permanent** (user-committed via C key, never evicted) and **temporary** (FIFO)
- `perm_end_pt` index marks the boundary — all eviction skips indices before it
- When full, oldest temporary tokens are compressed into long-term memory

### Tier 3: Long-term Memory (KeyValueMemoryStore)
- Compressed prototypes from evicted working memory
- Consolidation: top-k most-attended tokens become prototype keys, values are soft-readout through them
- Pruned by usage when exceeding `max_num_tokens`

### Permanent memory (C key / commit button)
- `MainController.on_commit()` → `processor.step(..., force_permanent=True)`
- Encodes frame image+mask into key/value tokens
- Stored with `as_permanent='all'` — prepended to front of tensor, `perm_end_pt` advanced
- Never evicted by FIFO, compression, or `clear_non_permanent_memory()`
- Only removed by "Clear all memory" button
- Use case: commit evenly-spaced keyframes (e.g., 0, 25, 50, 75, 100) as persistent anchors to prevent drift

---

## Comparison: ATLAS-Interactive vs SAM Family

### SAM 1 (Meta, early 2023)
- Single-image only, no video/temporal capability
- Prompt-based: points, boxes, masks
- No memory mechanism
- Class-agnostic instance segmentation

### SAM 2 / SAM 2.1 (Meta, July-September 2024)
- Extends SAM to video with streaming memory
- Memory bank: prompted frames + recent frames (FIFO, ~6 frames)
- No permanent memory, no memory compression, no long-term consolidation
- Old frames just drop off — drift on long videos
- SAM 2.1: better handling of similar objects and occlusions, torch.compile support
- Apache 2.0, actively maintained (github.com/facebookresearch/sam2, 18.7K stars)

### SAM 3 (Meta, November 2025)
- 848M parameters, dual encoder-decoder (DETR detector + SAM 2 tracker)
- Adds text prompts ("segment all graspers") and image exemplars
- Open-vocabulary concept segmentation — finds all instances of a concept
- Inherits SAM 2's memory bank (no tiered memory)
- Shared Perception Encoder improves re-identification after occlusions
- 30ms per image with 100+ objects on H200
- Submitted to ICLR 2026

### Key differences for surgical annotation

| | ATLAS (CUTIE+RITM) | SAM 2/2.1 | SAM 3 |
|---|---|---|---|
| Memory | 3-tier with permanent anchors | Flat FIFO bank | Flat FIFO bank + shared PE |
| Long video drift | Prevented by permanent + compressed LT memory | Drifts as bank drops old frames | Better re-ID but same bank |
| Segmentation type | Semantic (42 named classes) | Class-agnostic instances | Open-vocabulary concepts |
| Annotation | Clicks → RITM, then CUTIE propagates | Clicks/boxes, auto-propagates | Clicks/boxes/text, auto-propagates |
| Memory control | User chooses what to commit permanently | Implicit only | Implicit only |

CUTIE's permanent memory is the main advantage for long surgical videos with the commit-keyframes workflow.

---

## Newer Models Worth Considering (2025-2026)

### Surgical-specific (most relevant)

| Model | Year | What it does | Repo |
|---|---|---|---|
| **SurgiSAM2** | 2025 | SAM 2 fine-tuned on 5 surgical datasets (CholecSeg8k, Dresden, Endoscapes, etc.). +17.9% over baseline SAM 2 on surgical anatomy, outperforms SOTA in 80% of anatomy classes. | github.com/Devanish31/SurgiSAM2 |
| **MA-SAM2** | MICCAI 2025 | Training-free memory enhancement for SAM 2. Mask-quality-based memory selection for surgical videos. +6.1% on EndoVis2018. | github.com/Fawke108/MA-SAM2 |
| **MedSAM2** | 2025 | SAM 2 fine-tuned on 455K+ medical image-mask pairs + 76K video frames. Reduces annotation cost >85%. | github.com/bowang-lab/MedSAM2 |
| **ReSurgSAM2** | MICCAI 2025 | Mamba detection + SAM 2 tracking + diversity-driven long-term memory. 61.2 FPS. Supports text-referred segmentation. | github.com/jinlab-imvr/ReSurgSAM2 |
| **SASVi** | IPCAI 2025 | SAM 2 + object detection "Overseer" for automatic re-prompting when instruments enter/leave scene. | github.com/MECLabTUDA/SASVi |

### General VOS

| Model | Year | What it does | Repo |
|---|---|---|---|
| **SAM2Long** | ICCV 2025 | Training-free tree memory + object-aware memory modulation for SAM 2. +5.3 J&F on long-term benchmarks. Fixes SAM 2's drift on long videos. | github.com/Mark12Ding/SAM2Long |
| **SeC** | ICLR 2026 | SAM 2.1 + large vision-language model for concept-level reasoning. +11.8 J&F over SAM 2.1. Handles disappearance/reappearance via semantic understanding. | github.com/OpenIXCLab/SeC |
| **LiVOS** | CVPR 2025 | Linear attention memory — constant memory regardless of video length. 53% less GPU than STM methods. 4096p on 32GB. | github.com/uncbiag/LiVOS |

### RITM replacements (click-based single-frame segmentation)

| Model | Year | Improvement over RITM | Repo |
|---|---|---|---|
| **SimpleClick (ViT-H)** | ICCV 2023 | 20-30% fewer clicks, same codebase lineage — easiest drop-in | github.com/uncbiag/SimpleClick |
| **SAM 2.1 Tiny/Small** | Sep 2024 | Unified image+video, 81.7 mIoU@5clicks, Apache 2.0 | github.com/facebookresearch/sam2 |
| **HQ-SAM** | NeurIPS 2023 | Much better mask boundaries for intricate structures | github.com/SysCV/sam-hq |
| **FocalClick-XL** | Jun 2025 | Current SOTA on click benchmarks, supports clicks+scribbles+boxes | arxiv.org/abs/2506.14686 |
| **CFR-ICL** | AAAI 2024 | Built on RITM codebase, 33% fewer clicks on Berkeley | github.com/TitorX/CFR-ICL-Interactive-Segmentation |
| **ScribblePrompt** | ECCV 2024 | Medical-specific, 28% faster annotation, supports scribbles | github.com/halleewong/ScribblePrompt |

### Recommended upgrade path

1. **Replace RITM + CUTIE with SAM 2.1** — unified model for click interaction and temporal propagation
2. **Use SurgiSAM2 weights** — already fine-tuned on surgical anatomy
3. **Add SAM2Long at inference time** — training-free fix for long-video drift
4. **Re-implement permanent memory / commit workflow** on top of SAM 2's memory bank (GUI-level feature)

This would be a significant rewrite of the inference pipeline. For incremental wins, the config optimizations above give substantial quality improvement with minimal code changes.
