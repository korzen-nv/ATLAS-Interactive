"""Auto-segmentation backend using SurgNetXL (MetaFormerFPN) seg head.

Provides prompt-free surgical scene segmentation with 8 classes.
Designed to bootstrap masks for the interactive propagation backends.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

# ── 8-class SurgNetXL segmentation scheme ───────────────────────────────────
NUM_CLASSES = 8
CLASS_NAMES = [
    "Background",        # 0
    "Abdominal Wall",    # 1
    "Liver",             # 2
    "Gallbladder",       # 3
    "Fat",               # 4
    "Connective Tissue", # 5
    "Instruments",       # 6
    "Other Anatomy",     # 7
]

# Default mapping: SurgNetXL class ID → ATLAS CholecSeg object ID
# Adjust via config `autoseg_class_map` to match your palette.
DEFAULT_CLASS_MAP: Dict[int, int] = {
    0: 0,   # Background   → background
    1: 3,   # Abdominal Wall → 3 (Abdominal wall)
    2: 1,   # Liver         → 1 (Liver)
    3: 7,   # Gallbladder   → 7 (Gallbladder)
    4: 5,   # Fat           → 5 (Fat)
    5: 8,   # Connective T. → 8 (Connective tissue / Cystic plate)
    6: 11,  # Instruments   → 11 (Monopolar hook — first instrument slot)
    7: 15,  # Other Anatomy → 15 (Other structure)
}

_preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SurgNetSegBackend:
    """Wraps the SurgNetXL 8-class seg head for single-frame inference."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        class_map: Optional[Dict[int, int]] = None,
        *,
        torch_compile: bool = False,
        torch_compile_mode: str = "max-autotune-no-cudagraphs",
        fp8: bool = False,
    ) -> None:
        self._device = device
        self._class_map = class_map or DEFAULT_CLASS_MAP
        self._model = self._load_model(checkpoint, device)

        # GPU optimisation ---------------------------------------------------
        if fp8:
            try:
                from torchao.quantization import quantize_, Float8WeightOnlyConfig
                log.info("SurgNetSeg: applying FP8 weight-only quantization")
                quantize_(self._model, Float8WeightOnlyConfig())
            except ImportError:
                log.warning("fp8: torchao not installed — skipping.  "
                            "Install with: pip install torchao")
        if torch_compile:
            log.info("SurgNetSeg: torch.compile (mode=%s)", torch_compile_mode)
            self._model = torch.compile(
                self._model, mode=torch_compile_mode, dynamic=True)
        # --------------------------------------------------------------------

        log.info(
            "SurgNetSeg loaded (%.1fM params) on %s",
            sum(p.numel() for p in self._model.parameters()) / 1e6,
            device,
        )

    # ── public API ──────────────────────────────────────────────────────────

    @torch.inference_mode()
    def segment(self, image_np: np.ndarray) -> np.ndarray:
        """Run segmentation on an (H, W, 3) uint8 BGR/RGB numpy image.

        Returns:
            (H, W) uint8 array with *remapped* ATLAS object IDs.
        """
        orig_h, orig_w = image_np.shape[:2]
        pil_img = Image.fromarray(image_np)
        x = _preprocess(pil_img).unsqueeze(0).to(self._device)

        logits = self._model(x)  # (1, 8, 512, 512)
        pred = logits.argmax(dim=1).squeeze(0)  # (512, 512)

        # resize to original resolution
        pred = (
            F.interpolate(
                pred.float().unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="nearest",
            )
            .squeeze()
            .byte()
            .cpu()
            .numpy()
        )

        # remap SurgNetXL class IDs → ATLAS object IDs
        return self._remap(pred)

    # ── internals ───────────────────────────────────────────────────────────

    def _remap(self, pred: np.ndarray) -> np.ndarray:
        """Apply class_map LUT: SurgNetXL ID → ATLAS object ID."""
        lut = np.zeros(NUM_CLASSES, dtype=np.uint8)
        for src, dst in self._class_map.items():
            if src < NUM_CLASSES:
                lut[src] = dst
        return lut[pred]

    @staticmethod
    def _load_model(checkpoint: str, device: str):
        # MetaFormerFPN lives outside this repo.  Search known locations.
        home = Path.home()
        candidates = [
            home / "Projects" / "surg-diff" / "surgnet",   # primary
            home / "Projects" / "surgnet",
            Path(checkpoint).resolve().parent,              # next to checkpoint
        ]
        for p in candidates:
            d = str(p)
            if d not in sys.path and p.is_dir() and (p / "metaformer.py").exists():
                sys.path.insert(0, d)
                break

        from metaformer import MetaFormerFPN

        model = MetaFormerFPN(
            num_classes=NUM_CLASSES, pretrained="SurgeNet", pretrained_weights=None
        )
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        model.load_state_dict(state)
        model.to(device).eval()
        return model
