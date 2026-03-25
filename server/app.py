import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from omegaconf import DictConfig, open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from gui.cutie.utils.palette import custom_palette
from server.session import SessionManager

log = logging.getLogger(__name__)

# Global state shared across the app
app_state = {
    "cfg": None,
    "device": None,
    "cutie_model": None,
    "session_manager": None,
}


def load_config(images=None, video=None, workspace=None, num_objects=None) -> DictConfig:
    """Load Hydra config and merge runtime arguments."""
    GlobalHydra.instance().clear()
    initialize(version_base='1.3.2', config_path="../gui/cutie/config", job_name="web")
    cfg = compose(config_name="gui_config")

    if num_objects is None:
        num_objects = len(custom_palette) // 3 - 1

    with open_dict(cfg):
        cfg['images'] = images
        cfg['video'] = video
        cfg['workspace'] = workspace
        cfg['num_objects'] = num_objects
        cfg['workspace_init_only'] = False

    return cfg


def detect_device() -> str:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_models(cfg: DictConfig, device: str):
    """Load CUTIE and RITM model weights. Returns the CUTIE model."""
    from gui.cutie.model.cutie import CUTIE
    from gui.cutie.utils.download_models import download_models_if_needed

    download_models_if_needed()

    cutie = CUTIE(cfg).eval().to(device)
    model_weights = torch.load(cfg.weights, map_location=device)
    cutie.load_weights(model_weights)

    return cutie


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, clean up on shutdown."""
    torch.set_grad_enabled(False)

    device = detect_device()
    log.info(f"Using device: {device}")

    cfg = load_config()
    with open_dict(cfg):
        cfg['device'] = device
        cfg['amp'] = device == 'cuda'

    cutie_model = load_models(cfg, device)

    app_state["cfg"] = cfg
    app_state["device"] = device
    app_state["cutie_model"] = cutie_model
    app_state["session_manager"] = SessionManager()

    log.info("Models loaded, server ready.")
    yield

    # Cleanup
    app_state["cutie_model"] = None
    if device == 'cuda':
        torch.cuda.empty_cache()
    log.info("Server shut down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ATLAS-Interactive Web",
        description="Web-based interactive video labeling for surgical segmentation",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from server.routes_rest import router as rest_router
    from server.routes_ws import router as ws_router

    app.include_router(rest_router, prefix="/api")
    app.include_router(ws_router)

    # Serve frontend static files (built React app)
    frontend_dist = Path(__file__).parent.parent / "web" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
