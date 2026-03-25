import os
import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import Response, FileResponse, JSONResponse
from omegaconf import open_dict

from gui.cutie.utils.palette import custom_names, custom_palette_np
from server.frame_encoder import encode_frame_jpeg, encode_mask_png

router = APIRouter()


def get_state():
    from server.app import app_state
    return app_state


def get_session_manager():
    return get_state()["session_manager"]


def get_active_controller():
    sm = get_session_manager()
    session = sm.active_session
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")
    return session.controller


# ── Session management ────────────────────────────────────────────────

@router.post("/session")
async def create_session(
    workspace: str = Query(None),
    video: str = Query(None),
    images: str = Query(None),
):
    """Create a new session with a workspace."""
    state = get_state()
    cfg = state["cfg"].copy()
    with open_dict(cfg):
        cfg['workspace'] = workspace
        cfg['video'] = video
        cfg['images'] = images
        cfg['device'] = state['device']
        cfg['amp'] = state['device'] == 'cuda'

    from server.web_controller import WebController
    controller = WebController(cfg, state["cutie_model"])

    session_id = get_session_manager().create_session(controller)
    return {
        "session_id": session_id,
        "workspace": controller.cfg['workspace'],
        "total_frames": controller.length,
        "width": controller.w,
        "height": controller.h,
        "num_objects": controller.num_objects,
    }


@router.delete("/session/{session_id}")
async def destroy_session(session_id: str):
    if not get_session_manager().destroy_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok"}


@router.get("/sessions")
async def list_sessions():
    return get_session_manager().list_sessions()


# ── Workspace ─────────────────────────────────────────────────────────

@router.post("/workspace/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file, create workspace, and start session."""
    state = get_state()

    # Save uploaded file to temp location
    upload_dir = Path(state["cfg"]["workspace_root"]) / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / file.filename

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create session with this video
    cfg = state["cfg"].copy()
    with open_dict(cfg):
        cfg['video'] = str(video_path)
        cfg['images'] = None
        cfg['workspace'] = None
        cfg['device'] = state['device']
        cfg['amp'] = state['device'] == 'cuda'

    from server.web_controller import WebController
    controller = WebController(cfg, state["cutie_model"])
    session_id = get_session_manager().create_session(controller)

    return {
        "session_id": session_id,
        "workspace": controller.cfg['workspace'],
        "total_frames": controller.length,
        "width": controller.w,
        "height": controller.h,
    }


@router.post("/workspace/upload-images")
async def upload_images(file: UploadFile = File(...)):
    """Upload a zip of images, extract, create workspace, and start session."""
    import zipfile
    state = get_state()

    upload_dir = Path(state["cfg"]["workspace_root"]) / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    zip_path = upload_dir / file.filename

    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract to temp folder
    extract_dir = upload_dir / file.filename.rsplit('.', 1)[0]
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    cfg = state["cfg"].copy()
    with open_dict(cfg):
        cfg['images'] = str(extract_dir)
        cfg['video'] = None
        cfg['workspace'] = None
        cfg['device'] = state['device']
        cfg['amp'] = state['device'] == 'cuda'

    from server.web_controller import WebController
    controller = WebController(cfg, state["cutie_model"])
    session_id = get_session_manager().create_session(controller)

    return {
        "session_id": session_id,
        "workspace": controller.cfg['workspace'],
        "total_frames": controller.length,
        "width": controller.w,
        "height": controller.h,
    }


@router.get("/workspace/info")
async def workspace_info():
    ctrl = get_active_controller()
    return {
        "workspace": ctrl.cfg['workspace'],
        "total_frames": ctrl.length,
        "width": ctrl.w,
        "height": ctrl.h,
        "num_objects": ctrl.num_objects,
        "frame_names": [ctrl.res_man.names[i] for i in range(ctrl.length)],
    }


# ── Frame / Mask retrieval ────────────────────────────────────────────

@router.get("/frame/{ti}")
async def get_frame(ti: int, vis_mode: str = Query(None)):
    """Get a frame as JPEG, optionally with visualization overlay."""
    ctrl = get_active_controller()
    if ti < 0 or ti >= ctrl.length:
        raise HTTPException(status_code=404, detail="Frame index out of range")

    image = ctrl.res_man.get_image(ti)

    if vis_mode:
        from gui.interactive_utils import get_visualization
        mask = ctrl.res_man.get_mask(ti)
        if mask is None:
            import numpy as np
            mask = np.zeros((ctrl.h, ctrl.w), dtype=np.uint8)
        image = get_visualization(vis_mode, image, mask, ctrl.overlay_layer, ctrl.vis_target_objects)

    jpeg_bytes = encode_frame_jpeg(image)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/frame/{ti}/mask")
async def get_mask(ti: int):
    """Get a mask as PNG."""
    ctrl = get_active_controller()
    if ti < 0 or ti >= ctrl.length:
        raise HTTPException(status_code=404, detail="Frame index out of range")

    mask = ctrl.res_man.get_mask(ti)
    if mask is None:
        raise HTTPException(status_code=404, detail="No mask for this frame")

    png_bytes = encode_mask_png(mask)
    return Response(content=png_bytes, media_type="image/png")


# ── Config ────────────────────────────────────────────────────────────

@router.get("/config")
async def get_config():
    ctrl = get_active_controller()
    return {
        "mem_every": ctrl.cfg.get('mem_every', 5),
        "work_mem_min": ctrl.cfg.long_term.get('min_mem_frames', 5),
        "work_mem_max": ctrl.cfg.long_term.get('max_mem_frames', 10),
        "long_mem_max": ctrl.cfg.long_term.get('max_num_tokens', 10000),
        "output_fps": ctrl.output_fps,
        "output_bitrate": ctrl.output_bitrate,
        "vis_mode": ctrl.vis_mode,
    }


@router.put("/config")
async def update_config(
    work_mem_min: int = Query(None),
    work_mem_max: int = Query(None),
    long_mem_max: int = Query(None),
    mem_every: int = Query(None),
):
    ctrl = get_active_controller()
    await ctrl.update_config(work_mem_min, work_mem_max, long_mem_max, mem_every)
    return {"status": "ok"}


# ── Palette ───────────────────────────────────────────────────────────

@router.get("/palette")
async def get_palette():
    """Get class names and colors."""
    palette = []
    for obj_id, name in custom_names.items():
        r, g, b = custom_palette_np[obj_id]
        palette.append({
            "id": obj_id,
            "name": name,
            "color": [int(r), int(g), int(b)],
        })
    return palette


# ── Import ────────────────────────────────────────────────────────────

@router.post("/import/mask")
async def import_mask(file: UploadFile = File(...)):
    ctrl = get_active_controller()
    upload_dir = Path(ctrl.cfg['workspace']) / "_imports"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    await ctrl.import_mask(str(file_path))
    return {"status": "ok"}


@router.post("/import/layer")
async def import_layer(file: UploadFile = File(...)):
    ctrl = get_active_controller()
    upload_dir = Path(ctrl.cfg['workspace']) / "_imports"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    await ctrl.import_layer(str(file_path))
    return {"status": "ok"}


# ── Export ────────────────────────────────────────────────────────────

@router.post("/export/video")
async def export_video():
    ctrl = get_active_controller()
    output_path = ctrl.export_visualization()
    if output_path:
        return {"status": "ok", "path": output_path}
    raise HTTPException(status_code=404, detail="No visualization to export")


@router.post("/export/binary-masks")
async def export_binary():
    ctrl = get_active_controller()
    output_path = ctrl.export_binary()
    if output_path:
        return {"status": "ok", "path": output_path}
    raise HTTPException(status_code=404, detail="No masks to export")


@router.get("/export/download/{filename:path}")
async def download_export(filename: str):
    ctrl = get_active_controller()
    file_path = os.path.join(ctrl.cfg['workspace'], filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
