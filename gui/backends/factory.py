"""Factory for creating inference backends based on config."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig

from gui.backends.base import ClickBackend, PropagationBackend

log = logging.getLogger(__name__)


def create_auto_segmenter(cfg: DictConfig, device: str):
    """Create the SurgNetXL auto-segmentation backend if configured.

    Returns ``None`` when ``autoseg_weights`` is not set in config.
    """
    weights = cfg.get('autoseg_weights')
    if not weights:
        return None

    from gui.backends.surgnet_seg import SurgNetSegBackend

    # Build class map from config (list of "src:dst" ints) if provided
    class_map = None
    raw_map = cfg.get('autoseg_class_map')
    if raw_map:
        class_map = {int(k): int(v) for k, v in raw_map.items()}

    return SurgNetSegBackend(checkpoint=weights, device=device, class_map=class_map)


def create_backends(
    cfg: DictConfig,
    device: str,
    image_dir: Optional[str] = None,
    *,
    shared_model=None,
) -> Tuple[PropagationBackend, ClickBackend]:
    """Create propagation + click backends based on ``cfg.backend``.

    Args:
        cfg: Hydra config with ``backend`` key (default ``'cutie'``).
        device: ``'cuda'``, ``'mps'``, or ``'cpu'``.
        image_dir: Path to workspace JPEG frames directory (required for SAM).
        shared_model: Pre-loaded model object to avoid reloading weights
            (used by the web server where the model is loaded once at startup).

    Returns:
        ``(propagation_backend, click_backend)`` tuple.
    """
    backend_name = cfg.get('backend', 'cutie')
    log.info(f"Initialising backend: {backend_name}")

    if backend_name == 'cutie':
        return _create_cutie(cfg, device, shared_model)
    elif backend_name == 'sam2':
        return _create_sam2(cfg, device, image_dir, shared_model)
    elif backend_name == 'sam3':
        return _create_sam3(cfg, device, image_dir, shared_model)
    elif backend_name == 'sam31':
        return _create_sam31(cfg, device, image_dir, shared_model)
    else:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Expected one of: cutie, sam2, sam3, sam31"
        )


# -- CUTIE + RITM -------------------------------------------------------------

def _create_cutie(cfg, device, shared_model):
    from gui.backends.cutie_backend import CutieBackend, RitmClickBackend
    from gui.cutie.utils.download_models import download_models_if_needed

    if shared_model is not None:
        cutie = shared_model
    else:
        from gui.cutie.model.cutie import CUTIE
        download_models_if_needed()
        cutie = CUTIE(cfg).eval().to(device)
        weights = torch.load(cfg.weights, map_location=device)
        cutie.load_weights(weights)

    propagation = CutieBackend(cutie, cfg)
    click = RitmClickBackend(cfg.ritm_weights, device=device)
    return propagation, click


# -- SAM 2 --------------------------------------------------------------------

def _create_sam2(cfg, device, image_dir, shared_model):
    from gui.backends.sam2_backend import Sam2ClickBackend, Sam2PropagationBackend

    checkpoint = cfg.get('sam2_weights')
    model_cfg = cfg.get('sam2_model_cfg')
    if not checkpoint:
        raise ValueError("sam2_weights must be set in config when using backend: sam2")

    propagation = Sam2PropagationBackend(
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        device=device,
        image_dir=image_dir,
        num_objects=cfg.num_objects,
        shared_predictor=shared_model,
    )
    click = Sam2ClickBackend(
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        device=device,
        shared_model=shared_model,
    )
    return propagation, click


# -- SAM 3 --------------------------------------------------------------------

def _create_sam3(cfg, device, image_dir, shared_model):
    from gui.backends.sam3_backend import Sam3ClickBackend, Sam3PropagationBackend

    checkpoint = cfg.get('sam3_weights')  # None → auto-download from HF
    bpe_path = cfg.get('sam3_bpe_path')

    propagation = Sam3PropagationBackend(
        checkpoint=checkpoint,
        bpe_path=bpe_path,
        device=device,
        image_dir=image_dir,
        num_objects=cfg.num_objects,
        shared_model=shared_model,
    )
    # The tracker is built without a vision backbone (it's on the detector).
    # SAM3InteractiveImagePredictor needs the backbone for forward_image(),
    # so share the detector's backbone with the tracker.
    tracker = propagation._model.tracker
    if tracker.backbone is None:
        tracker.backbone = propagation._model.detector.backbone
    click = Sam3ClickBackend(
        tracker_model=tracker,
        device=device,
    )
    return propagation, click


# -- SAM 3.1 (Multiplex) ------------------------------------------------------

def _create_sam31(cfg, device, image_dir, shared_model):
    from gui.backends.sam31_backend import Sam31ClickBackend, Sam31PropagationBackend

    checkpoint = cfg.get('sam31_weights')  # None → auto-download from HF
    lora_weights = cfg.get('sam31_lora_weights')  # None → no LoRA

    propagation = Sam31PropagationBackend(
        checkpoint=checkpoint,
        device=device,
        image_dir=image_dir,
        num_objects=cfg.num_objects,
        shared_model=shared_model,
        lora_weights=lora_weights,
    )
    click = Sam31ClickBackend(
        multiplex_model=propagation._model,
        device=device,
    )
    return propagation, click
