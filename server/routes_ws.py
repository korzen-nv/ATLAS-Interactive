import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
log = logging.getLogger(__name__)


def get_state():
    from server.app import app_state
    return app_state


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    state = get_state()
    sm = state["session_manager"]
    session = sm.get_session(session_id)

    if session is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    ctrl = session.controller

    # Set up the WebSocket send function
    async def ws_send(msg: dict):
        try:
            await websocket.send_json(msg)
        except Exception:
            pass  # connection may have closed

    ctrl.set_ws_send(ws_send)

    # Send initial state
    await ctrl.send_state()
    await ctrl.send_frame()
    await ctrl.send_memory_status()

    # Play video task handle
    play_task = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "click":
                await ctrl.click_fn(msg["action"], msg["x"], msg["y"])

            elif msg_type == "mouse_move":
                await ctrl.on_mouse_motion(msg["x"], msg["y"])

            elif msg_type == "navigate":
                await ctrl.navigate_to_frame(msg["frame"])
                await ctrl.send_state()

            elif msg_type == "propagate":
                direction = msg.get("direction", "forward")
                if direction == "forward":
                    asyncio.create_task(ctrl.on_forward_propagation())
                else:
                    asyncio.create_task(ctrl.on_backward_propagation())

            elif msg_type == "pause":
                await ctrl.pause_propagation()
                await ctrl.send_propagation_state()

            elif msg_type == "commit":
                await ctrl.on_commit()

            elif msg_type == "set_object":
                await ctrl.set_object(msg["id"])

            elif msg_type == "set_vis_mode":
                await ctrl.set_vis_mode(msg["mode"])
                await ctrl.send_state()

            elif msg_type == "toggle_vis_mode":
                await ctrl.toggle_vis_mode()

            elif msg_type == "reset_frame":
                await ctrl.on_reset_mask()

            elif msg_type == "reset_object":
                await ctrl.on_reset_object()

            elif msg_type == "clear_memory":
                if msg.get("permanent", True):
                    await ctrl.on_clear_memory()
                else:
                    await ctrl.on_clear_non_permanent_memory()

            elif msg_type == "play_video":
                playing = msg.get("playing", False)
                ctrl.playing = playing
                if playing:
                    async def play_loop():
                        while ctrl.playing:
                            await ctrl.play_video_tick()
                            await asyncio.sleep(1 / 30)
                    if play_task is None or play_task.done():
                        play_task = asyncio.create_task(play_loop())
                else:
                    ctrl.playing = False
                    await ctrl.send_state()

            elif msg_type == "update_config":
                await ctrl.update_config(
                    work_mem_min=msg.get("work_mem_min"),
                    work_mem_max=msg.get("work_mem_max"),
                    long_mem_max=msg.get("long_mem_max"),
                    mem_every=msg.get("mem_every"),
                )

            elif msg_type == "get_memory_status":
                await ctrl.send_memory_status()

            elif msg_type == "get_state":
                await ctrl.send_state()

            else:
                log.warning(f"Unknown WebSocket message type: {msg_type}")

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: session {session_id}")
    except Exception as e:
        log.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        ctrl.clear_ws_send()
        ctrl.playing = False
