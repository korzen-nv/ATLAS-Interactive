"""Factory for creating inference backends based on config."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig

from gui.backends.base import ClickBackend, PropagationBackend

log = logging.getLogger(__name__)


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
    else:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Expected one of: cutie, sam2, sam3"
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
    from gui.backends.sam3_backend import Sam3PropagationBackend
    from gui.backends.sam2_backend import Sam2ClickBackend

    checkpoint = cfg.get('sam3_weights')
    bpe_path = cfg.get('sam3_bpe_path')
    if not checkpoint:
        raise ValueError("sam3_weights must be set in config when using backend: sam3")

    propagation = Sam3PropagationBackend(
        checkpoint=checkpoint,
        bpe_path=bpe_path,
        device=device,
        image_dir=image_dir,
        num_objects=cfg.num_objects,
        shared_predictor=shared_model,
    )
    # SAM 3 is backward-compatible with SAM 2 for point/click interaction
    click = Sam2ClickBackend(
        checkpoint=checkpoint,
        model_cfg=cfg.get('sam3_model_cfg'),
        device=device,
        shared_model=shared_model,
    )
    return propagation, click
