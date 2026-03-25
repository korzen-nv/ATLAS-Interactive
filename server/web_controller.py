import os
from os import path
import asyncio
import logging
from typing import Literal, Optional, Callable, Any

import cv2
import torch
from torch import autocast
from torchvision.transforms.functional import to_tensor
import numpy as np
from omegaconf import DictConfig, open_dict

from gui.cutie.model.cutie import CUTIE
from gui.cutie.inference.inference_core import InferenceCore
from gui.interaction import ClickInteraction, aggregate_wbg
from gui.interactive_utils import (
    image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch,
    get_visualization, get_visualization_torch,
)
from gui.resource_manager import ResourceManager
from gui.click_controller import ClickController
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from gui.cutie.utils.palette import custom_palette_np, custom_names

from server.frame_encoder import encode_frame_base64

log = logging.getLogger(__name__)


class WebController:
    """
    Adapted from MainController — same business logic, but replaces all
    PySide6 GUI calls with WebSocket message emissions.
    """

    def __init__(self, cfg: DictConfig, cutie_model: CUTIE) -> None:
        self.initialized = False
        self._ws_send: Optional[Callable] = None
        self._loop = None
        self._message_queue: list = []

        # workspace setup
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"])
            else:
                raise ValueError('Either images, video, or workspace must be specified')
            cfg = cfg.copy()
            with open_dict(cfg):
                cfg["workspace"] = path.join(cfg['workspace_root'], basename)

        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # use the shared CUTIE model
        self.cutie = cutie_model

        # initialize RITM click controller
        self.click_ctrl = ClickController(cfg.ritm_weights, device=self.device)

        # main components
        self.res_man = ResourceManager(cfg)
        self.processor = InferenceCore(self.cutie, self.cfg)

        # control state
        self.length: int = self.res_man.length
        self.interaction = None
        self.interaction_type: str = 'Click'
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0

        # current frame
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros(
            (self.num_objects + 1, self.h, self.w), dtype=torch.float
        ).to(self.device)
        self.curr_prob[0] = 1

        # visualization
        self.vis_mode: str = 'davis'
        self.vis_image: np.ndarray = None
        self.save_visualization_mode: str = 'None'
        self.save_soft_mask: bool = False
        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        # polygon mode
        self.polygon_points = []
        self.hover_first_point = False
        self.hover_threshold = 8
        self.in_polygon_mode = False

        # export settings
        self.output_fps = cfg['output_fps']
        self.output_bitrate = cfg['output_bitrate']

        # playing state
        self.playing = False

        # load first frame
        self.load_current_image_mask()
        self.compose_current_im()

        self.initialized = True

    # ── WebSocket bridge ──────────────────────────────────────────────

    def set_ws_send(self, ws_send: Callable):
        """Set the WebSocket send function. Messages queued before this are flushed."""
        self._ws_send = ws_send
        self._loop = asyncio.get_event_loop()
        for msg in self._message_queue:
            asyncio.ensure_future(ws_send(msg))
        self._message_queue.clear()

    def clear_ws_send(self):
        self._ws_send = None
        self._loop = None

    async def _send(self, msg: dict):
        if self._ws_send:
            await self._ws_send(msg)
        else:
            self._message_queue.append(msg)

    def _send_sync(self, msg: dict):
        """Non-async send for use in synchronous code paths (e.g. from executor threads)."""
        if self._ws_send and self._loop:
            asyncio.run_coroutine_threadsafe(self._ws_send(msg), self._loop)
        else:
            self._message_queue.append(msg)

    async def send_console(self, text: str):
        await self._send({"type": "console", "text": text})

    async def send_frame(self):
        """Encode and send the current visualization frame."""
        if self.vis_image is None:
            return
        image_b64 = encode_frame_base64(self.vis_image)
        await self._send({
            "type": "frame",
            "ti": self.curr_ti,
            "image": image_b64,
            "name": self.res_man.names[self.curr_ti] + '.jpg',
            "total_frames": self.length,
        })

    async def send_state(self):
        """Send full UI state snapshot."""
        await self._send({
            "type": "state",
            "current_frame": self.curr_ti,
            "total_frames": self.length,
            "current_object": self.curr_object,
            "vis_mode": self.vis_mode,
            "propagating": self.propagating,
            "propagate_direction": self.propagate_direction,
            "polygon_mode": self.in_polygon_mode,
            "polygon_points": self.polygon_points,
            "playing": self.playing,
            "frame_name": self.res_man.names[self.curr_ti] + '.jpg',
            "width": self.w,
            "height": self.h,
        })

    async def send_memory_status(self):
        """Send memory gauge data."""
        data = {"type": "memory_status"}
        try:
            data["perm_tokens"] = self.processor.memory.work_mem.perm_size(0)
            data["work_tokens"] = self.processor.memory.work_mem.non_perm_size(0)
            data["max_work_tokens"] = self.processor.memory.max_work_tokens
            data["long_tokens"] = self.processor.memory.long_mem.non_perm_size(0)
            data["max_long_tokens"] = self.processor.memory.max_long_tokens
        except AttributeError:
            data["work_tokens"] = 0
            data["max_work_tokens"] = 1
            data["long_tokens"] = 0
            data["max_long_tokens"] = 1

        # GPU info
        if 'cuda' in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_total_gb = global_total / (2**30)
            global_used_gb = (global_total - global_free) / (2**30)
            torch_used_gb = torch.cuda.max_memory_allocated() / (2**30)
            data["gpu_used_gb"] = round(global_used_gb, 2)
            data["gpu_total_gb"] = round(global_total_gb, 2)
            data["torch_used_gb"] = round(torch_used_gb, 2)
        else:
            data["gpu_used_gb"] = 0
            data["gpu_total_gb"] = 0
            data["torch_used_gb"] = 0

        await self._send(data)

    async def send_progress(self, value: float):
        await self._send({"type": "progress", "value": value})

    async def send_propagation_state(self):
        await self._send({
            "type": "propagation_state",
            "propagating": self.propagating,
            "direction": self.propagate_direction,
        })

    async def send_polygon_update(self):
        await self._send({
            "type": "polygon_update",
            "points": self.polygon_points,
            "hover_first": self.hover_first_point,
            "polygon_mode": self.in_polygon_mode,
        })

    # ── Properties ────────────────────────────────────────────────────

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T

    # ── Frame management ──────────────────────────────────────────────

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti)
        self.curr_image_torch = None

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.curr_ti)
            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()
            self.curr_prob = None

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        if self.curr_image_torch is None:
            self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device, non_blocking=True)
        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_numpy_to_one_hot_torch(
                self.curr_mask, self.num_objects + 1
            ).to(self.device, non_blocking=True)

    def compose_current_im(self):
        self.vis_image = get_visualization(
            self.vis_mode, self.curr_image_np, self.curr_mask,
            self.overlay_layer, self.vis_target_objects
        )

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        self.vis_image = get_visualization_torch(
            self.vis_mode, self.curr_image_torch, self.curr_prob,
            self.overlay_layer_torch, self.vis_target_objects
        )
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        save_visualization = self.save_visualization_mode in [
            'Propagation only (higher quality)', 'Always'
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())

    async def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            if self.save_visualization_mode == 'Always':
                self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        await self.send_frame()

    def save_current_mask(self):
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

    # ── Click / Polygon interaction ───────────────────────────────────

    async def click_fn(self, action: Literal['left', 'right', 'middle'], x: int, y: int):
        if self.propagating:
            return

        if action == 'middle':
            self.in_polygon_mode = not self.in_polygon_mode
            self.polygon_points = []
            self.hover_first_point = False
            mode_text = 'Polygon mode ON' if self.in_polygon_mode else 'Click mode ON'
            await self.send_console(mode_text)
            self.compose_current_im()
            await self.send_frame()
            await self.send_polygon_update()
            return

        if self.in_polygon_mode:
            await self._polygon_click(action, x, y)
            return

        # Normal click interaction
        last_interaction = self.interaction
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if last_interaction is None or last_interaction.tar_obj != self.curr_object:
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    self.interaction = ClickInteraction(
                        image, self.curr_prob, (self.h, self.w),
                        self.click_ctrl, self.curr_object
                    )

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                await self._update_interacted_mask()

    async def _polygon_click(self, action: str, x: int, y: int):
        if action == 'left':
            if self.polygon_points:
                first_pt = self.polygon_points[0]
                dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
                if dist <= self.hover_threshold:
                    # Finalize polygon
                    if self.polygon_points[-1] != first_pt:
                        self.polygon_points.append(first_pt)

                    mask = np.zeros((self.h, self.w), dtype=np.uint8)
                    pts_np = np.array(
                        [[(int(px), int(py)) for px, py in self.polygon_points]],
                        dtype=np.int32
                    )
                    cv2.fillPoly(mask, pts_np, color=1)
                    self.curr_mask[mask > 0] = self.curr_object
                    self.save_current_mask()

                    self.curr_prob = index_numpy_to_one_hot_torch(
                        self.curr_mask, self.num_objects + 1
                    ).to(self.device)

                    self.polygon_points = []
                    self.hover_first_point = False
                    await self.show_current_frame()
                    await self.send_console('Polygon finalized and added to segmentation.')
                    await self.send_polygon_update()
                    return

            self.polygon_points.append((x, y))
            await self.send_console(f'Polygon point added: ({x}, {y})')
            self._compose_polygon_overlay()
            await self.send_frame()
            await self.send_polygon_update()

        elif action == 'right':
            if self.polygon_points:
                removed = self.polygon_points.pop()
                await self.send_console(f'Removed polygon point: {removed}')
                self._compose_polygon_overlay()
                await self.send_frame()
                await self.send_polygon_update()
            else:
                await self.send_console('No points to remove.')

    def _compose_polygon_overlay(self):
        self.compose_current_im()
        pts = [(int(px), int(py)) for (px, py) in self.polygon_points]
        r, g, b = custom_palette_np[self.curr_object]
        r, g, b = int(r), int(g), int(b)

        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(self.vis_image, pts[i], pts[i + 1], color=(r, g, b), thickness=1)

        for i, pt in enumerate(pts):
            if i == 0 and self.hover_first_point:
                color = (255, 255, 255)
                radius = 6
            else:
                color = (r, g, b)
                radius = 4
            cv2.circle(self.vis_image, pt, radius=radius, color=color, thickness=-1)

    async def on_mouse_motion(self, x: int, y: int):
        self.last_ex, self.last_ey = x, y
        if self.polygon_points:
            first_pt = self.polygon_points[0]
            dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
            was_hovering = self.hover_first_point
            self.hover_first_point = dist <= self.hover_threshold
            if self.hover_first_point != was_hovering:
                self._compose_polygon_overlay()
                await self.send_frame()
                await self.send_polygon_update()

    # ── Interaction helpers ───────────────────────────────────────────

    async def _update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        await self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    # ── Frame navigation ──────────────────────────────────────────────

    async def navigate_to_frame(self, ti: int):
        ti = max(0, min(ti, self.length - 1))
        if self.propagating:
            return

        if self.curr_frame_dirty:
            self.save_current_mask()
        self.curr_frame_dirty = False

        self.curr_ti = ti
        self.reset_this_interaction()
        self.load_current_image_mask()
        await self.show_current_frame()

    async def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        await self.navigate_to_frame(new_ti)

    async def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        await self.navigate_to_frame(new_ti)

    # ── Object selection ──────────────────────────────────────────────

    async def set_object(self, number: int):
        if number == self.curr_object:
            return
        number = max(1, min(number, self.num_objects))
        self.curr_object = number
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        await self.send_console(f'Current object changed to {number}.')
        await self.show_current_frame()
        await self.send_state()

    # ── Visualization ─────────────────────────────────────────────────

    async def set_vis_mode(self, mode: str):
        self.vis_mode = mode
        await self.show_current_frame()

    async def toggle_vis_mode(self):
        if self.vis_mode == 'davis':
            self.vis_mode = 'light'
        elif self.vis_mode == 'light':
            self.vis_mode = 'davis'
        else:
            self.vis_mode = 'davis'
        await self.show_current_frame()
        await self.send_state()

    # ── Propagation ───────────────────────────────────────────────────

    async def on_forward_propagation(self):
        if self.propagating:
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_direction = 'forward'
            await self.send_propagation_state()
            await self._propagate(forward=True)

    async def on_backward_propagation(self):
        if self.propagating:
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_direction = 'backward'
            await self.send_propagation_state()
            await self._propagate(forward=False)

    async def _propagate(self, forward: bool):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._propagate_sync, forward)
        # After propagation ends (back on async thread)
        self.propagating = False
        self.curr_frame_dirty = False
        self.propagate_direction = 'none'
        await self._send({"type": "console", "text": f'Propagation stopped at t={self.curr_ti}.'})
        await self._send({"type": "propagation_state", "propagating": False, "direction": "none"})

    def _propagate_sync(self, forward: bool):
        """Synchronous propagation loop run in executor thread."""
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self._send_sync({"type": "console", "text": f'Propagation started at t={self.curr_ti}.'})
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(
                self.curr_image_torch, self.curr_prob[1:], idx_mask=False
            )
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            self.interacted_prob = None
            self.reset_this_interaction()

            # Show initial frame (fast path)
            self.update_current_image_fast(invalid_soft_mask=True)
            self._send_sync({
                "type": "frame",
                "ti": self.curr_ti,
                "image": encode_frame_base64(self.vis_image),
                "name": self.res_man.names[self.curr_ti] + '.jpg',
                "total_frames": self.length,
            })

            self.propagating = True

            dataset = PropagationReader(
                self.res_man, self.curr_ti, 'forward' if forward else 'backward'
            )
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            for i, data in enumerate(loader):
                if not self.propagating:
                    break

                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)

                # Step frame index
                if forward:
                    self.curr_ti = min(self.curr_ti + 1, self.length - 1)
                else:
                    self.curr_ti = max(0, self.curr_ti - 1)

                self.curr_prob = self.processor.step(self.curr_image_torch)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                self.save_current_mask()
                self.update_current_image_fast()

                # Send frame update
                self._send_sync({
                    "type": "frame",
                    "ti": self.curr_ti,
                    "image": encode_frame_base64(self.vis_image),
                    "name": self.res_man.names[self.curr_ti] + '.jpg',
                    "total_frames": self.length,
                })

                # Send progress
                progress = (i + 1) / max(len(dataset), 1)
                self._send_sync({"type": "progress", "value": progress})

                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

    async def pause_propagation(self):
        self.propagating = False

    # ── Memory management ─────────────────────────────────────────────

    async def on_commit(self):
        if self.interacted_prob is None:
            self.load_current_image_mask()
        else:
            self.complete_interaction()
            await self._update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            await self.send_console(f'Permanent memory saved at {self.curr_ti}.')
            self.curr_prob = self.processor.step(
                self.curr_image_torch, self.curr_prob[1:],
                idx_mask=False, force_permanent=True
            )
            await self.send_memory_status()

    async def on_clear_memory(self):
        self.processor.clear_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        self.processor.update_config(self.cfg)
        await self.send_memory_status()
        await self.send_console('All memory cleared.')

    async def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        self.processor.update_config(self.cfg)
        await self.send_memory_status()
        await self.send_console('Non-permanent memory cleared.')

    # ── Reset ─────────────────────────────────────────────────────────

    async def on_reset_mask(self):
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        await self.show_current_frame()

    async def on_reset_object(self):
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        await self.show_current_frame()

    # ── Config ────────────────────────────────────────────────────────

    async def update_config(self, work_mem_min=None, work_mem_max=None,
                            long_mem_max=None, mem_every=None):
        with open_dict(self.cfg):
            if work_mem_min is not None:
                self.cfg.long_term['min_mem_frames'] = work_mem_min
            if work_mem_max is not None:
                self.cfg.long_term['max_mem_frames'] = work_mem_max
            if long_mem_max is not None:
                self.cfg.long_term['max_num_tokens'] = long_mem_max
            if mem_every is not None:
                self.cfg['mem_every'] = mem_every
        self.processor.update_config(self.cfg)
        await self.send_console('Config updated.')

    # ── Import / Export ───────────────────────────────────────────────

    async def import_mask(self, file_path: str):
        mask = self.res_man.import_mask(file_path, size=(self.h, self.w))
        shape_ok = len(mask.shape) == 2 and mask.shape[-1] == self.w and mask.shape[-2] == self.h
        object_ok = mask.max() <= self.num_objects

        if not shape_ok:
            await self.send_console(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
        elif not object_ok:
            await self.send_console(f'Expected {self.num_objects} objects. Got {mask.max()} instead.')
        else:
            await self.send_console(f'Mask loaded from {file_path}.')
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            await self.show_current_frame()
            self.save_current_mask()

    async def import_layer(self, file_path: str):
        try:
            layer = self.res_man.import_layer(file_path, size=(self.h, self.w))
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            await self.send_console(f'Layer loaded from {file_path}.')
            await self.show_current_frame()
        except FileNotFoundError:
            await self.send_console(f'{file_path} not found.')

    def export_visualization(self, progress_callback=None):
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            convert_frames_to_video(
                image_folder, output_path,
                fps=self.output_fps, bitrate=self.output_bitrate,
                progress_callback=progress_callback
            )
            return output_path
        return None

    def export_binary(self, progress_callback=None):
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        save_folder = path.join(self.cfg['workspace'], 'binary_masks')
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            convert_mask_to_binary(
                mask_folder, save_folder, self.vis_target_objects,
                progress_callback=progress_callback
            )
            return save_folder
        return None

    # ── Play video ────────────────────────────────────────────────────

    async def play_video_tick(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.load_current_image_mask(no_mask=True)
        self.compose_current_im()
        await self.send_frame()
