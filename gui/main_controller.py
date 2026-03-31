import os
from collections import deque
from os import path
import logging
from typing import Literal

import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

from scipy.ndimage import binary_dilation

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
import numpy as np
from omegaconf import DictConfig, open_dict

from gui.backends.factory import create_auto_segmenter, create_backends

from gui.interaction import *
from gui.interactive_utils import *
from gui.resource_manager import ResourceManager
from gui.gui import GUI
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from gui.global_memory import GlobalMemoryStore

from gui.cutie.utils.palette import custom_palette_np # added

log = logging.getLogger()



class MainController():

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.initialized = False

        # setting up the workspace
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = path.basename(cfg["video"]) #[:-4]
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            cfg["workspace"] = path.join(cfg['workspace_root'], basename)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # main components
        self.res_man = ResourceManager(cfg)
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            return

        # persistent cross-video memory
        workspace_root = cfg.get('workspace_root', os.path.dirname(cfg['workspace']))
        global_mem_dir = cfg.get('global_memory_dir',
                                 os.path.join(workspace_root, 'global_memory'))
        self.global_memory = GlobalMemoryStore(global_mem_dir)

        # initializing the network(s) — after ResourceManager so image_dir is available
        self.initialize_networks()
        self.processor = self._propagation
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
        self.interaction_type: str = 'Click'
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros((self.num_objects + 1, self.h, self.w),
                                                   dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        # undo stack — stores (frame_idx, mask, prob) snapshots before each edit
        self._undo_stack: deque = deque(maxlen=20)

        # track which frames have been committed to permanent memory
        self.permanent_memory_frames: set[int] = set()

        # visualization info
        self.vis_mode: str = 'davis'
        self.vis_image: np.ndarray = None
        self.save_visualization_mode: str = 'None'
        self.save_soft_mask: bool = False
        self.fill_gaps: bool = False

        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        self.load_current_image_mask()
        self.show_current_frame()

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()
        if hasattr(self.processor, 'memory'):
            # CUTIE-specific memory tuning spinboxes
            self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
            self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
            self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
            self.gui.mem_every_box.setValue(self.processor.mem_every)

        # for exporting videos
        self.output_fps = cfg['output_fps']
        self.output_bitrate = cfg['output_bitrate']

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn

        # Variables for polygon drawing and hovering first point
        self.polygon_points = []
        self.hover_first_point = False
        self.hover_threshold = 8  # pixels
        self.in_polygon_mode = False

        self.gui.show()
        self._refresh_global_memory_list()
        self.gui.text('Initialized.')
        self.initialized = True

        # try to load the default overlay
        self._try_load_layer('./docs/uiuc.png')
        self.gui.set_object_color(self.curr_object)
        self.update_config()

    def initialize_networks(self) -> None:
        self._propagation, self.click_ctrl = create_backends(
            self.cfg, self.device, image_dir=self.res_man.image_dir,
        )
        self._auto_seg = None  # lazy-loaded on first use

    def hit_number_key(self, number: int):
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_object_color(number)
        self.gui.set_current_object_id(number)
        self.show_current_frame()
    
    def on_mouse_motion_xy(self, x: int, y: int):
        # Check if polygon is being drawn and at least one point exists
        if self.polygon_points:
            # Check distance to first point
            first_pt = self.polygon_points[0]
            dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
            was_hovering = self.hover_first_point
            self.hover_first_point = dist <= self.hover_threshold
            
            # If hover state changed, redraw polygon overlay to update color
            if self.hover_first_point != was_hovering:
                self.compose_polygon_overlay()
                self.update_canvas()
        self.last_ex, self.last_ey = x, y

    def compose_polygon_overlay(self):
        # Reset to base visualization image
        self.compose_current_im()

        # Draw polygon points and lines
        pts = [(int(px), int(py)) for (px, py) in self.polygon_points]

        # Get color for the current object
        r, g, b = custom_palette_np[self.curr_object] 
        r, g, b = int(r), int(g), int(b)

        # Draw lines between points
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(self.vis_image, pts[i], pts[i + 1], color=(r,g,b), thickness=1)

        # Draw points with hover effect on first point
        for i, pt in enumerate(pts):
            if i == 0 and self.hover_first_point:
                # Hover color: white
                color = (255, 255, 255)
                radius = 6
            else:
                # Normal color: yellow
                color = (r,g,b)
                radius = 4
            cv2.circle(self.vis_image, pt, radius=radius, color=color, thickness=-1)

    def click_fn(self, action: Literal['left', 'right', 'middle', 'pick'], x: int, y: int):
        if self.propagating:
            return

        if action == 'pick':
            obj_id = int(self.curr_mask[int(y), int(x)])
            if obj_id > 0 and obj_id != self.curr_object:
                self.hit_number_key(obj_id)
            elif obj_id == 0:
                self.gui.text('No object at this position.')
            return

        if not hasattr(self, 'in_polygon_mode'):
            self.in_polygon_mode = False  # new flag to track current mode

        if action == 'middle':
            # Toggle polygon mode
            self.in_polygon_mode = not self.in_polygon_mode
            self.polygon_points = []
            self.hover_first_point = False
            mode_text = 'Polygon mode ON' if self.in_polygon_mode else 'Click mode ON'
            self.gui.text(mode_text)
            self.compose_current_im()
            self.update_canvas()
            return

        if self.in_polygon_mode:
            # In polygon drawing mode
            if action == 'left':
                if self.polygon_points:
                    first_pt = self.polygon_points[0]
                    dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
                    if dist <= self.hover_threshold:
                        # Finalize polygon
                        self.gui.text('Finalizing polygon.')

                        # Close polygon loop
                        if self.polygon_points[-1] != first_pt:
                            self.polygon_points.append(first_pt)

                        # Create binary mask
                        mask = np.zeros((self.h, self.w), dtype=np.uint8)
                        pts_np = np.array([[(int(px), int(py)) for px, py in self.polygon_points]], dtype=np.int32)
                        cv2.fillPoly(mask, pts_np, color=1)
                        self._snapshot_mask()
                        self.curr_mask[mask > 0] = self.curr_object
                        self.save_current_mask()

                        # ✅ Update probability map so it's used for propagation
                        self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(self.device)

                        self.polygon_points = []
                        self.hover_first_point = False
                        self.show_current_frame()
                        self.gui.text('Polygon finalized and added to segmentation.')
                        return
                # Add new point
                self.polygon_points.append((x, y))
                self.gui.text(f'Polygon point added: ({x}, {y})')
                self.compose_polygon_overlay()
                self.update_canvas()
                return

            elif action == 'right':
                # Remove last point
                if self.polygon_points:
                    removed = self.polygon_points.pop()
                    self.gui.text(f'Removed polygon point: {removed}')
                    self.compose_polygon_overlay()
                    self.update_canvas()
                else:
                    self.gui.text('No points to remove.')
                return

            else:
                # Do nothing in polygon mode for other actions
                return

        # Not in polygon mode: do normal click interaction
        last_interaction = self.interaction
        new_interaction = None

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if (last_interaction is None or last_interaction.tar_obj != self.curr_object):
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    new_interaction = ClickInteraction(image, self.curr_prob, (self.h, self.w),
                                                    self.click_ctrl, self.curr_object)
                    if new_interaction is not None:
                        self.interaction = new_interaction

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                self.update_interacted_mask()
                self.update_gpu_gauges()

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
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

    def compose_current_im(self):
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, self.curr_mask,
                                           self.overlay_layer, self.vis_target_objects)

    def update_canvas(self):
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # thus current_image_torch must be voided afterwards
        # do_no_save_soft_mask is an override to solve #41
        self.vis_image = get_visualization_torch(self.vis_mode, self.curr_image_torch,
                                                 self.curr_prob, self.overlay_layer_torch,
                                                 self.vis_target_objects)
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        save_visualization = self.save_visualization_mode in [
            'Propagation only (higher quality)', 'Always'
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            if self.save_visualization_mode == 'Always':
                self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
            self.update_canvas()

        self.gui.update_slider(self.curr_ti)
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + '.jpg')

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def set_vis_mode_direct(self, mode: str):
        self.vis_mode = mode
        self.gui.combo.setCurrentText(mode)
        self.show_current_frame()

    def _snapshot_mask(self):
        """Push the current mask state onto the undo stack (before mutation)."""
        prob = self.curr_prob.cpu().clone() if self.curr_prob is not None else None
        self._undo_stack.append((self.curr_ti, self.curr_mask.copy(), prob))

    def on_undo(self):
        if self.propagating or not self._undo_stack:
            return
        ti, mask, prob = self._undo_stack.pop()
        self.curr_ti = ti
        self.curr_mask = mask
        self.curr_prob = prob.to(self.device) if prob is not None else None
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()
        self.gui.update_slider(self.curr_ti)
        self.gui.text('Undo.')

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

    def on_slider_update(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        self.curr_ti = self.gui.tl_slider.value()
        if not self.propagating:
            # with self.vis_cond:
            #     self.vis_cond.notify()
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False

            self.reset_this_interaction()
            self.curr_ti = self.gui.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.on_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = 'backward'
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f'Propagation stopped at t={self.curr_ti}.')
        self.gui.pause_propagation()

    def on_propagate(self):
        # start to propagate
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.processor.clear_sensory_memory()
            if hasattr(self.processor, 'set_propagation_direction'):
                self.processor.set_propagation_direction(
                    self.propagate_direction == 'backward')
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False,
                                                 frame_idx=self.curr_ti)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            if self.fill_gaps:
                self.curr_mask = self.fill_mask_gaps(self.curr_mask)
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            # override this for #41
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # propagate till the end
            for data in loader:
                if not self.propagating:
                    break
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()

                self.curr_prob = self.processor.step(self.curr_image_torch,
                                                     frame_idx=self.curr_ti)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
                if self.fill_gaps:
                    self.curr_mask = self.fill_mask_gaps(self.curr_mask)

                self.save_current_mask()
                self.show_current_frame(fast=True)

                self.update_memory_gauges()
                self.gui.process_events()

                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

            self.propagating = False
            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def on_propagate_forward_one(self, steps=1):
        if self.propagating or self.curr_ti >= self.T - 1:
            return
        self._propagate_n_frames('forward', steps)

    def on_propagate_backward_one(self, steps=1):
        if self.propagating or self.curr_ti <= 0:
            return
        self._propagate_n_frames('backward', steps)

    def _propagate_n_frames(self, direction, n=1):
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            self.processor.clear_sensory_memory()
            if hasattr(self.processor, 'set_propagation_direction'):
                self.processor.set_propagation_direction(direction == 'backward')
            self.processor.step(self.curr_image_torch,
                                self.curr_prob[1:],
                                idx_mask=False,
                                frame_idx=self.curr_ti)

            for _ in range(n):
                at_boundary = (direction == 'forward' and self.curr_ti >= self.T - 1) or \
                              (direction == 'backward' and self.curr_ti <= 0)
                if at_boundary:
                    break

                if direction == 'forward':
                    self.on_next_frame()
                else:
                    self.on_prev_frame()

                self.curr_image_np = self.res_man.get_image(self.curr_ti)
                self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device)

                self.curr_prob = self.processor.step(self.curr_image_torch,
                                                      frame_idx=self.curr_ti)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
                if self.fill_gaps:
                    self.curr_mask = self.fill_mask_gaps(self.curr_mask)

                self.save_current_mask()
                self.show_current_frame(fast=True)
                self.gui.process_events()

            self.curr_frame_dirty = False
            self.show_current_frame()
            self.update_memory_gauges()
            self.update_gpu_gauges()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False,
                                                 frame_idx=self.curr_ti,
                                                 force_permanent=True)
            self.permanent_memory_frames.add(self.curr_ti)
            self.gui.tl_slider.set_markers(self.permanent_memory_frames)
            self.update_memory_gauges()
            self.update_gpu_gauges()

    def on_play_video_timer(self):
        self.curr_ti += 1
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            self.gui.text(f'Exporting visualization -- please wait')
            self.gui.process_events()
            convert_frames_to_video(image_folder,
                                    output_path,
                                    fps=self.output_fps,
                                    bitrate=self.output_bitrate,
                                    progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Visualization exported to {output_path}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No visualization images found in {image_folder}')

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        save_folder = path.join(self.cfg['workspace'], 'binary_masks')
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text(f'Exporting binary masks -- please wait')
            self.gui.process_events()
            convert_mask_to_binary(mask_folder,
                                   save_folder,
                                   self.vis_target_objects,
                                   progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Binary masks exported to {save_folder}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No masks found in {mask_folder}')

    def on_object_dial_change(self):
        object_id = self.gui.object_dial.value()
        self.hit_number_key(object_id)

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

    def update_interacted_mask(self):
        self._snapshot_mask()
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        self._snapshot_mask()
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_reset_object(self):
        self._snapshot_mask()
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_remove_object_all_frames(self):
        """Remove the current object's segmentation from every frame in the workspace."""
        if self.propagating:
            return

        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self.gui, 'Remove object from all frames',
            f'Remove object {self.curr_object} from ALL {self.T} frames?\n'
            'This cannot be undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        obj_id = self.curr_object
        self.gui.text(f'Removing object {obj_id} from all frames...')
        self.gui.process_events()

        modified = 0
        for ti in range(self.T):
            mask = self.res_man.get_mask(ti)
            if mask is None:
                continue
            if not np.any(mask == obj_id):
                continue
            mask = mask.copy()
            mask[mask == obj_id] = 0
            self.res_man.save_mask(ti, mask)
            modified += 1

            if modified % 50 == 0:
                self.gui.progressbar_update(ti / self.T)
                self.gui.process_events()

        self.gui.progressbar_update(0)

        # Reload current frame to reflect changes
        self.load_current_image_mask()
        self.curr_prob = None
        self.reset_this_interaction()
        self.show_current_frame()

        self.gui.text(f'Object {obj_id} removed from {modified} frame(s).')

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_marker(self):
        after = sorted(i for i in self.permanent_memory_frames if i > self.curr_ti)
        if after:
            self.gui.tl_slider.setValue(after[0])

    def on_prev_marker(self):
        before = sorted((i for i in self.permanent_memory_frames if i < self.curr_ti), reverse=True)
        if before:
            self.gui.tl_slider.setValue(before[0])

    def update_gpu_gauges(self):
        if 'cuda' in self.device:
            try:
                info = torch.cuda.mem_get_info()
                global_free, global_total = info
                global_free /= (2**30)
                global_total /= (2**30)
                global_used = global_total - global_free

                self.gui.gpu_mem_gauge.setFormat(f'{global_used:.1f} GB / {global_total:.1f} GB')
                self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

                used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
                self.gui.torch_mem_gauge.setFormat(f'{used_by_torch:.1f} GB / {global_total:.1f} GB')
                self.gui.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100))
            except torch.cuda.CudaError:
                self.gui.gpu_mem_gauge.setFormat('OOM')
                self.gui.gpu_mem_gauge.setValue(100)
                self.gui.torch_mem_gauge.setFormat('OOM')
                self.gui.torch_mem_gauge.setValue(100)
        elif 'mps' in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f'{mem_used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            status = self.processor.get_memory_status()

            self.gui.perm_mem_gauge.setFormat(f'{status.perm_tokens} / {status.perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100 if status.perm_tokens else 0)

            self.gui.work_mem_gauge.setFormat(f'{status.work_tokens} / {status.max_work_tokens}')
            self.gui.work_mem_gauge.setValue(
                round(status.work_tokens / max(status.max_work_tokens, 1) * 100))

            self.gui.long_mem_gauge.setFormat(f'{status.long_tokens} / {status.max_long_tokens}')
            self.gui.long_mem_gauge.setValue(
                round(status.long_tokens / max(status.max_long_tokens, 1) * 100))

        except AttributeError:
            self.gui.work_mem_gauge.setFormat('Unknown')
            self.gui.long_mem_gauge.setFormat('Unknown')
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(),
                    self.gui.work_mem_max.value() - 1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(),
                    self.gui.work_mem_min.value() + 1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term['min_mem_frames'] = self.gui.work_mem_min.value()
                self.cfg.long_term['max_mem_frames'] = self.gui.work_mem_max.value()
                self.cfg.long_term['max_num_tokens'] = self.gui.long_mem_max.value()
                self.cfg['mem_every'] = self.gui.mem_every_box.value()

            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        self.permanent_memory_frames.clear()
        self.gui.tl_slider.set_markers(self.permanent_memory_frames)
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_import_mask(self):
        file_name = self.gui.open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))

        shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w)
                           and (mask.shape[-2] == self.h))

        object_condition = (mask.max() <= self.num_objects)

        if not shape_condition:
            self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
        elif not object_condition:
            self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.gui.text(f'Mask file {file_name} loaded.')
            self._snapshot_mask()
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            self.show_current_frame()
            self.save_current_mask()

    def on_import_layer(self):
        file_name = self.gui.open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f'Layer file {file_name} loaded.')
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f'{file_name} not found.')

    def on_auto_segment(self):
        """Run SurgNetXL auto-segmentation on the current frame."""
        if self.propagating:
            return

        # Lazy-load the auto-seg model on first use
        if self._auto_seg is None:
            self.gui.text('Loading SurgNetXL seg head...')
            self.gui.process_events()
            self._auto_seg = create_auto_segmenter(self.cfg, self.device)
            if self._auto_seg is None:
                self.gui.text('Auto-seg not configured (set autoseg_weights in config).')
                return

        self.gui.text(f'Auto-segmenting frame {self.curr_ti}...')
        self.gui.process_events()

        # Run inference on the current frame
        seg_mask = self._auto_seg.segment(self.curr_image_np)

        # Validate
        if seg_mask.max() > self.num_objects:
            self.gui.text(
                f'Auto-seg produced class {seg_mask.max()} but num_objects={self.num_objects}. '
                f'Check autoseg_class_map in config.'
            )
            return

        # Snapshot for undo, then apply
        self._snapshot_mask()
        self.curr_mask = seg_mask
        self.curr_prob = index_numpy_to_one_hot_torch(
            self.curr_mask, self.num_objects + 1
        ).to(self.device)

        self.save_current_mask()
        self.show_current_frame()
        self.gui.text(f'Auto-segmented frame {self.curr_ti}.')

    # -- Global memory (cross-video exemplars) ----------------------------------

    def on_save_to_global_memory(self):
        """Save the current frame's image + mask to the persistent global store."""
        if self.curr_mask.max() == 0:
            self.gui.text('Nothing to save — current mask is empty.')
            return

        video_name = os.path.basename(self.cfg['workspace'])
        frame_name = self.res_man.names[self.curr_ti]

        objects = sorted(set(int(v) for v in np.unique(self.curr_mask)) - {0})
        item = self.global_memory.add(
            name=frame_name,
            image=self.curr_image_np,
            mask=self.curr_mask,
            source_video=video_name,
            frame_idx=self.curr_ti,
            palette=self.res_man.palette,
        )
        self._refresh_global_memory_list()
        self.gui.text(
            f'Saved frame {self.curr_ti} to {video_name}/ — '
            f'objects {objects}, '
            f'{self.global_memory.total_count()} total items.'
        )

    def on_save_all_to_global_memory(self):
        """Save all permanent-memory frames to global memory, overwriting existing."""
        if not self.permanent_memory_frames:
            self.gui.text('No permanent memory frames to save.')
            return

        video_name = os.path.basename(self.cfg['workspace'])

        # remove existing folder for this video to avoid duplicates
        self.global_memory.remove_folder(video_name)

        saved = 0
        for ti in sorted(self.permanent_memory_frames):
            image_np = self.res_man.get_image(ti)
            mask_np = self.res_man.get_mask(ti)
            if mask_np is None or mask_np.max() == 0:
                self.gui.text(f'  Skipping frame {ti} — empty mask.')
                continue
            frame_name = self.res_man.names[ti]
            self.global_memory.add(
                name=frame_name,
                image=image_np,
                mask=mask_np,
                source_video=video_name,
                frame_idx=ti,
                palette=self.res_man.palette,
            )
            saved += 1

        self._refresh_global_memory_list()
        self.gui.text(
            f'Saved {saved} frame(s) to global memory ({video_name}/).'
        )

    def _load_markers_from_global_memory(self):
        """If global memory has a folder matching this workspace, restore timeline markers."""
        video_name = os.path.basename(self.cfg['workspace'])
        items = self.global_memory.folder_items(video_name)
        if not items:
            return
        for item in items:
            if 0 <= item.frame_idx < self.length:
                self.permanent_memory_frames.add(item.frame_idx)
        if self.permanent_memory_frames:
            self.gui.tl_slider.set_markers(self.permanent_memory_frames)
            self.gui.text(
                f'Loaded {len(self.permanent_memory_frames)} marker(s) from global memory ({video_name}/).'
            )

    # -- loading from global memory -------------------------------------------

    def _inject_items(self, items):
        """Inject a list of GlobalMemoryItems into permanent memory."""
        loaded = 0
        for item in items:
            self.gui.text(f'  Loading "{item.name}" from {item.source_video} (frame {item.frame_idx})...')
            self.gui.process_events()

            foreign_image_np = item.load_image()
            foreign_mask_np = item.load_mask()

            if foreign_mask_np.shape[:2] != (self.h, self.w):
                foreign_mask_np = cv2.resize(foreign_mask_np, (self.w, self.h),
                                             interpolation=cv2.INTER_NEAREST)
            if foreign_image_np.shape[:2] != (self.h, self.w):
                foreign_image_np = cv2.resize(foreign_image_np, (self.w, self.h),
                                              interpolation=cv2.INTER_LINEAR)

            if foreign_mask_np.max() > self.num_objects:
                self.gui.text(
                    f'  Skipped "{item.name}" — class {foreign_mask_np.max()} '
                    f'exceeds num_objects={self.num_objects}.'
                )
                continue

            foreign_image_torch = to_tensor(foreign_image_np).to(self.device)
            foreign_mask_torch = torch.from_numpy(
                foreign_mask_np.astype(np.int64)
            ).to(self.device)
            objects = sorted(set(int(v) for v in np.unique(foreign_mask_np)) - {0})

            with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
                self.processor.inject_permanent_memory(
                    foreign_image_torch, foreign_mask_torch, objects,
                )
            loaded += 1
            self.gui.text(f'  Done — objects {objects}, shape {foreign_image_np.shape[:2]}')

        self.update_memory_gauges()
        return loaded

    def _selected_folder_name(self):
        """Return the folder name selected in the global memory list, or None."""
        rows = [idx.row() for idx in self.gui.global_mem_list.selectedIndexes()]
        if not rows:
            return None
        folders = self.global_memory.folders()
        row = rows[0]
        if row >= len(folders):
            return None
        return folders[row][0]

    def on_load_folder_from_global_memory(self):
        """Inject all items from the selected folder."""
        folder_name = self._selected_folder_name()
        if folder_name is None:
            self.gui.text('Select a folder first.')
            return

        items = self.global_memory.folder_items(folder_name)
        n = self._inject_items(items)
        self._load_markers_from_global_memory()
        self.gui.text(f'Injected {n} item(s) from {folder_name}.')

    def on_load_all_from_global_memory(self):
        """Inject all items from all folders."""
        items = self.global_memory.all_items()
        if not items:
            self.gui.text('Global memory is empty.')
            return
        n = self._inject_items(items)
        self._load_markers_from_global_memory()
        self.gui.text(f'Injected {n} item(s) from global memory.')

    def on_remove_global_memory_folder(self):
        """Remove the selected folder from the global memory store."""
        folder_name = self._selected_folder_name()
        if folder_name is None:
            self.gui.text('Select a folder to remove.')
            return
        self.global_memory.remove_folder(folder_name)
        self._refresh_global_memory_list()
        self.gui.text(f'Removed folder {folder_name}.')

    def _refresh_global_memory_list(self):
        video_name = os.path.basename(self.cfg['workspace'])
        self.gui.update_global_memory_list(self.global_memory.folders(), video_name)

    def _open_in_explorer(self, path):
        import subprocess, sys
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif sys.platform == 'win32':
            subprocess.Popen(['explorer', path])
        else:
            subprocess.Popen(['xdg-open', path])

    def on_open_workspace(self):
        """Open the current workspace folder in the system file manager."""
        workspace = self.cfg['workspace']
        self._open_in_explorer(workspace)
        self.gui.text(f'Opening {workspace}')

    def on_open_global_memory_folder(self, item):
        """Open the double-clicked global memory folder in the file manager."""
        folder_name = item.text().rsplit('(', 1)[0].strip()
        folder_path = str(self.global_memory.store_dir / folder_name)
        self._open_in_explorer(folder_path)
        self.gui.text(f'Opening {folder_path}')

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_fill_gaps_toggle(self):
        self.fill_gaps = self.gui.fill_gaps_checkbox.isChecked()
        state = 'ON' if self.fill_gaps else 'OFF'
        self.gui.text(f'Fill gaps: {state}')

    def fill_mask_gaps(self, mask: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Dilate each labeled segment into background pixels to close thin gaps."""
        bg_mask = (mask == 0)
        if not bg_mask.any():
            return mask
        mask = mask.copy()
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids > 0]
        for obj_id in obj_ids:
            obj_mask = (mask == obj_id)
            dilated = binary_dilation(obj_mask, iterations=iterations)
            fill = dilated & bg_mask
            mask[fill] = obj_id
            bg_mask[fill] = False
        return mask

    def on_mouse_motion_xy(self, x: int, y: int):
        self.last_ex, self.last_ey = x, y

        # Check if polygon is being drawn and at least one point exists
        if self.polygon_points:
            # Check distance to first point
            first_pt = self.polygon_points[0]
            dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
            was_hovering = self.hover_first_point
            self.hover_first_point = dist <= self.hover_threshold

            # If hover state changed, update the canvas
            if self.hover_first_point != was_hovering:
                self.compose_polygon_overlay()
                self.update_canvas()

    def on_toggle_vis_mode(self):
        if self.vis_mode == 'davis':
            self.vis_mode = 'light'
        elif self.vis_mode == 'light':
            self.vis_mode = 'image'
        elif self.vis_mode == 'image':
            self.vis_mode = 'davis'
        else:
            self.vis_mode = 'davis'
        print(f'Visualization mode changed to {self.vis_mode}')

        # Update the dropdown menu to show the current mode
        self.gui.combo.setCurrentText(self.vis_mode)
        self.show_current_frame()

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T
