"""Change-point detection using SurgeNetXL frame embeddings.

Extracts per-frame embeddings with CaFormer S18 (SurgeNetXL weights),
computes cosine distance between consecutive frames, and identifies
peaks in the distance signal as change points.  Also produces per-location
spatial distance maps showing WHERE each change occurs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from torchvision import transforms

log = logging.getLogger(__name__)

SURGNETXL_URL = (
    "https://huggingface.co/TimJaspersTue/SurgeNetModels/resolve/main/"
    "SurgeNetXL_checkpoint_epoch0050_teacher.pth?download=true"
)


def _resolve_surgnet_path() -> None:
    """Add the SurgeNet repo to sys.path so ``from metaformer import ...`` works.

    Mirrors the search logic in ``gui/backends/surgnet_seg.py``.
    """
    home = Path.home()
    candidates = [
        home / "Projects" / "surg-diff" / "surgnet",
        home / "Projects" / "surgnet",
    ]
    for p in candidates:
        d = str(p)
        if d not in sys.path and p.is_dir() and (p / "metaformer.py").exists():
            sys.path.insert(0, d)
            return
    log.warning("Could not find SurgeNet repo (metaformer.py) in known locations")


def detect_change_points(
    res_man,
    total_frames: int,
    device: str,
    subsample: int = 1,
    input_size: int = 256,
    sigma: float = 2.0,
    prominence: float = 0.3,
    distance: int = 10,
    progress_callback: Optional[Callable[[float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Detect change-point frames using SurgeNetXL embeddings.

    Args:
        res_man: ResourceManager with ``get_image(ti)`` returning (H,W,3) uint8 numpy.
        total_frames: Number of frames in the video.
        device: Torch device string ('cuda', 'cpu', etc.).
        subsample: Process every Nth frame (1 = all frames).
        input_size: Resize dimension for SurgeNet input.
        sigma: Gaussian smoothing sigma for the distance signal.
        prominence: Peak detection prominence threshold (lower = more sensitive).
        distance: Minimum distance between peaks in *subsampled* index space.
        progress_callback: Called with float 0.0-1.0 for progress updates.
        cancel_check: Called each iteration; return True to abort.

    Returns:
        Tuple of:
        - Sorted list of frame indices where significant changes were detected.
        - Dict mapping each change frame index to a (16, 16) spatial distance map.
    """
    # ── load model ──────────────────────────────────────────────────────────
    _resolve_surgnet_path()
    from metaformer import caformer_s18

    model = caformer_s18(
        num_classes=12,
        pretrained="SurgeNet",
        pretrained_weights=SURGNETXL_URL,
    )
    model.to(device).eval()
    log.info(
        "Change detector: CaFormer S18 loaded (%.1fM params) on %s",
        sum(p.numel() for p in model.parameters()) / 1e6,
        device,
    )

    # ── preprocessing (matches surgnet_seg.py convention) ───────────────────
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── extract embeddings + stage-3 features ───────────────────────────────
    frame_indices = list(range(0, total_frames, subsample))
    num_frames = len(frame_indices)
    embeddings: list[torch.Tensor] = []
    stage3_features: list[torch.Tensor] = []

    try:
        with torch.inference_mode():
            for i, ti in enumerate(frame_indices):
                if cancel_check and cancel_check():
                    log.info("Change detection cancelled at frame %d", ti)
                    return [], {}

                image_np = res_man.get_image(ti)
                pil_img = Image.fromarray(image_np)
                x = preprocess(pil_img).unsqueeze(0).to(device)

                global_emb, feature_list = model.forward_features(x)
                embeddings.append(global_emb.squeeze(0).cpu())
                # stage 3: (1, 320, 16, 16) at 256 input
                stage3_features.append(feature_list[2].squeeze(0).cpu())

                if progress_callback and i % 10 == 0:
                    progress_callback(i / num_frames)

        if progress_callback:
            progress_callback(0.95)

        if len(embeddings) < 2:
            return [], {}

        # ── compute frame-to-frame cosine distance ──────────────────────────
        emb = torch.stack(embeddings)  # (N, 512)
        distances = 1.0 - F.cosine_similarity(emb[:-1], emb[1:])  # (N-1,)
        distances = distances.numpy()

        # ── smooth and detect peaks ─────────────────────────────────────────
        if sigma > 0:
            smoothed = gaussian_filter1d(distances, sigma=sigma)
        else:
            smoothed = distances

        peaks, _ = find_peaks(smoothed, prominence=prominence, distance=distance)

        # ── compute spatial distance maps for change frames ─────────────────
        spatial_maps: dict[int, np.ndarray] = {}
        for p in peaks:
            if p + 1 >= len(frame_indices):
                continue
            change_ti = int(frame_indices[p + 1])
            feat_a = stage3_features[p].unsqueeze(0)      # (1, 320, 16, 16)
            feat_b = stage3_features[p + 1].unsqueeze(0)   # (1, 320, 16, 16)
            spatial_dist = 1.0 - F.cosine_similarity(feat_a, feat_b, dim=1)
            spatial_maps[change_ti] = spatial_dist.squeeze(0).numpy()  # (16, 16)

        # free stage-3 features
        del stage3_features

        change_frames = sorted(spatial_maps.keys())

        if progress_callback:
            progress_callback(1.0)

        log.info("Change detection found %d change points", len(change_frames))
        return change_frames, spatial_maps

    finally:
        # free GPU memory
        del model
        if device != "cpu":
            torch.cuda.empty_cache()
