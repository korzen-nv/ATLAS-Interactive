# ATLAS-Interactive Usage Guide

Applies to all backends (CUTIE, SAM 2, SAM 3).

## Core Workflow

### 1. Pick a good starting frame
Choose a frame where all objects are clearly visible and not occluded. This is your first anchor.

### 2. Annotate each object
- Select object 1 (press **1**)
- **Left-click** inside the object (positive click) -- a single well-placed click in the center often gives a good initial mask
- **Right-click** on false-positive areas to push the boundary back
- Switch to object 2 (press **2**), repeat
- You'll see the mask update in real-time after each click

Fewer, better-placed clicks beat many scattered clicks. Start with 1-2 positive clicks in the center, then refine with negatives only where the boundary is wrong.

### 3. Commit the anchor
Press **C** to save this frame to permanent memory. The model propagates *from* committed anchors.

### 4. Propagate
Press **F** (or Space) to propagate forward. The model will auto-segment every subsequent frame using your anchor.

### 5. Correct drift with new anchors
This is where the real quality comes from:
- Scrub through the results (arrow keys or slider)
- When you spot a frame where the mask drifted, stop there
- Refine with clicks, press **C** to commit a new anchor
- Press **F** again to re-propagate from this new anchor
- For frames *before* your anchor, press **B** for backward propagation

## Tips for Best Quality

**Anchor placement strategy** -- Don't just anchor frame 0 and hope. Place anchors at:
- First frame where an object appears
- Frames right after occlusions end (object reappears)
- Points where objects change shape significantly (e.g., instrument opens/closes)
- Every ~50-100 frames for long videos, even if things look fine

**Negative clicks matter** -- If the model bleeds into a neighboring structure, one well-placed right-click on the false region does more than adding extra positive clicks.

**Polygon mode** (middle-click to toggle) -- Use this for thin/elongated structures that are hard to capture with point clicks. Draw a tight polygon around the object.

**Multi-object order** -- Annotate all objects on a frame *before* committing. The commit saves the combined mask for all objects at once.

**Backward propagation** -- If you annotate frame 50, you can press **B** to propagate backward to frame 0. Useful when the first few frames are harder to annotate than a later keyframe.

## CUTIE-Specific: Memory Tuning

CUTIE exposes four memory controls (not available in SAM 2/3):
- **Min. working memory frames** -- minimum anchor frames kept in working memory
- **Max. working memory frames** -- cap on working memory size
- **Max. long-term memory size** -- cap on long-term memory tokens
- **Memory frame every (r)** -- how often (in frames) a new memory frame is stored during propagation

Lower values of *r* give more frequent memory updates (better quality, more VRAM). Increase if you run out of GPU memory on long videos.

## Quick Reference

| Key | Action |
|-----|--------|
| Left-click | Positive click (foreground) |
| Right-click | Negative click (background) |
| Middle-click | Toggle polygon mode |
| 1-9 | Select object |
| C | Commit to permanent memory |
| F / Space | Forward propagation |
| B | Backward propagation |
| Left/Right | Prev/next frame |
| Shift+Left/Right | Jump 10 frames |
| T | Toggle visualization mode |
