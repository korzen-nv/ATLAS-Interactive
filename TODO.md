# TODO — Potential Improvements

## Interaction
- [ ] **Brush/Eraser Drag** — Fix mouse drag painting (currently requires individual clicks)
- [ ] **Superpixel-Guided Annotation** — SLIC/Felzenszwalb superpixels for fast region selection
- [ ] **Multi-Scale / Zoom Interaction** — Crop-and-segment at full resolution for fine structures (instrument tips, small vessels)

## Visualization
- [ ] **Edge Detection Overlay** — Toggle-able Canny/Sobel edge map to reveal tissue boundaries in low-contrast areas

## Propagation QA
- [ ] **Optical Flow Validation** — RAFT/GMFlow to detect large motions, warp masks as sanity check, flag divergent frames

## Performance
- [ ] **ONNX/TensorRT Export** — Convert backends for faster per-frame inference during propagation

## Export
- [ ] **Mask Format Export** — COCO JSON, NIfTI, DICOM-SEG for downstream pipelines and clinical review

## Analytics
- [ ] **Annotation Analytics Dashboard** — Per-frame annotation time, clicks per object, propagation success rate
