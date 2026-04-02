import functools
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from PySide6.QtWidgets import (QWidget, QComboBox, QCheckBox, QHBoxLayout, QLabel, QPushButton,
                               QTextEdit, QSpinBox, QPlainTextEdit, QVBoxLayout, QSizePolicy,
                               QButtonGroup, QSlider, QRadioButton, QApplication, QFileDialog,
                               QListWidget, QListWidgetItem, QScrollArea, QFrame, QDoubleSpinBox,
                               QGroupBox)

from PySide6.QtGui import (QKeySequence, QShortcut, QTextCursor, QImage, QPixmap, QIcon, QPainter,
                            QColor, QPolygonF)
from PySide6.QtCore import Qt, QTimer, QSize, QKeyCombination, QPointF

from gui.cutie.utils.palette import custom_palette_np, custom_names
from gui.gui_utils import *
from gui.ritm import controller


class MarkerSlider(QSlider):
    """QSlider subclass that draws markers on the groove for permanent memory frames."""

    HANDLE_W = 8
    MARKER_R = 4
    MARKER_R_HIGHLIGHT = 6
    CHANGE_R = 3
    CHANGE_R_HIGHLIGHT = 5
    HIT_RADIUS = 8  # px tolerance for hover/click detection

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._markers: set[int] = set()
        self._marker_color = QColor(0, 200, 255)
        self._marker_active_color = QColor(255, 255, 100)
        self._change_markers: set[int] = set()
        self._change_color = QColor(255, 160, 0)
        self._change_active_color = QColor(255, 200, 100)
        self._uncertainty_markers: set[int] = set()
        self._uncertainty_color = QColor(255, 50, 50)
        self._uncertainty_active_color = QColor(255, 150, 80)
        self._hovered_idx: int | None = None
        self.setMouseTracking(True)

    def set_markers(self, frame_indices: set[int]):
        self._markers = set(frame_indices)
        self.update()

    def set_change_markers(self, frame_indices: set[int]):
        self._change_markers = set(frame_indices)
        self.update()

    def clear_change_markers(self):
        self._change_markers.clear()
        self.update()

    def set_uncertainty_markers(self, frame_indices: set[int]):
        self._uncertainty_markers = set(frame_indices)
        self.update()

    def clear_uncertainty_markers(self):
        self._uncertainty_markers.clear()
        self.update()

    def _groove_params(self):
        groove_left = self.HANDLE_W
        groove_width = self.width() - 2 * self.HANDLE_W
        return groove_left, groove_width

    def _idx_to_x(self, idx):
        span = self.maximum() - self.minimum()
        if span == 0:
            return None
        groove_left, groove_width = self._groove_params()
        return groove_left + (idx - self.minimum()) / span * groove_width

    def _x_to_nearest_marker(self, x):
        best_idx, best_dist = None, self.HIT_RADIUS + 1
        for idx in self._markers | self._change_markers | self._uncertainty_markers:
            mx = self._idx_to_x(idx)
            if mx is None:
                continue
            dist = abs(x - mx)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def mouseMoveEvent(self, event):
        old = self._hovered_idx
        self._hovered_idx = self._x_to_nearest_marker(event.position().x())
        if old != self._hovered_idx:
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hovered_idx is not None:
            self._hovered_idx = None
            self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        # always let QSlider finish its drag state first
        super().mouseReleaseEvent(event)
        # if clicking on a marker (permanent or change), jump to that frame
        if event.button() == Qt.MouseButton.LeftButton and (self._markers or self._change_markers or self._uncertainty_markers):
            clicked = self._x_to_nearest_marker(event.position().x())
            if clicked is not None:
                self.setValue(clicked)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._markers and not self._change_markers and not self._uncertainty_markers:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        groove_y = self.height() // 2
        current_val = self.value()

        # draw uncertainty markers (red upward triangles)
        for idx in self._uncertainty_markers:
            mx = self._idx_to_x(idx)
            if mx is None:
                continue
            active = (idx == self._hovered_idx or idx == current_val)
            r = self.CHANGE_R_HIGHLIGHT if active else self.CHANGE_R
            color = self._uncertainty_active_color if active else self._uncertainty_color
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            triangle = QPolygonF([
                QPointF(mx, groove_y - r),
                QPointF(mx + r, groove_y + r),
                QPointF(mx - r, groove_y + r),
            ])
            painter.drawPolygon(triangle)

        # draw change markers (below permanent markers)
        for idx in self._change_markers:
            mx = self._idx_to_x(idx)
            if mx is None:
                continue
            active = (idx == self._hovered_idx or idx == current_val)
            r = self.CHANGE_R_HIGHLIGHT if active else self.CHANGE_R
            color = self._change_active_color if active else self._change_color
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            diamond = QPolygonF([
                QPointF(mx, groove_y - r),
                QPointF(mx + r, groove_y),
                QPointF(mx, groove_y + r),
                QPointF(mx - r, groove_y),
            ])
            painter.drawPolygon(diamond)

        # draw permanent memory markers on top (circles)
        for idx in self._markers:
            mx = self._idx_to_x(idx)
            if mx is None:
                continue
            active = (idx == self._hovered_idx or idx == current_val)
            r = self.MARKER_R_HIGHLIGHT if active else self.MARKER_R
            color = self._marker_active_color if active else self._marker_color
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(int(mx) - r, groove_y - r, r * 2, r * 2)

        painter.end()


class GUI(QWidget):

    def __init__(self, controller, cfg: DictConfig) -> None:
        super().__init__()

        # callbacks to be set by the controller
        self.on_mouse_motion_xy = None
        self.click_fn = None
        self.on_mouse_release_fn = lambda: None
        self._mouse_buttons = None

        self.controller = controller
        self.cfg = cfg
        self.h = controller.h
        self.w = controller.w
        self.T = controller.T

        # set up the window
        self.setWindowTitle(f'SurgeNetSeg demo: {cfg["workspace"]}')
        self.setGeometry(100, 100, self.w + 200, self.h + 200)
        self.setWindowIcon(QIcon('docs/icon.png'))

        # set up some buttons
        self.play_button = QPushButton('Play video')
        self.play_button.clicked.connect(self.on_play_video)
        self.play_x4_button = QPushButton('Play x4')
        self.play_x4_button.clicked.connect(self.on_play_video_x4)
        self.commit_button = QPushButton('Commit to permanent memory')
        self.commit_button.clicked.connect(controller.on_commit)

        self.forward_run_button = QPushButton('Propagate forward')
        self.forward_run_button.clicked.connect(controller.on_forward_propagation)
        self.forward_run_button.setMinimumWidth(150)

        self.backward_run_button = QPushButton('Propagate backward')
        self.backward_run_button.clicked.connect(controller.on_backward_propagation)
        self.backward_run_button.setMinimumWidth(150)

        # gap-filling toggle
        self.fill_gaps_checkbox = QCheckBox('Fill gaps')
        self.fill_gaps_checkbox.setChecked(False)
        self.fill_gaps_checkbox.setToolTip(
            'Dilate each segment into unassigned (background) pixels to close thin gaps between segments'
        )
        self.fill_gaps_checkbox.stateChanged.connect(controller.on_fill_gaps_toggle)

        # CRF refinement toggle
        self.crf_checkbox = QCheckBox('CRF refine')
        self.crf_checkbox.setChecked(False)
        self.crf_checkbox.setToolTip('Apply Dense CRF to snap mask boundaries to image edges')
        self.crf_checkbox.stateChanged.connect(controller.on_crf_toggle)

        # soft mask saving toggle
        self.save_soft_mask_checkbox = QCheckBox('Save soft masks')
        self.save_soft_mask_checkbox.setChecked(False)
        self.save_soft_mask_checkbox.setToolTip('Save per-object soft probability masks alongside hard masks')
        self.save_soft_mask_checkbox.stateChanged.connect(controller.on_save_soft_mask_toggle)

        # universal progressbar
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(0)
        self.progressbar.setMinimumWidth(200)

        self.reset_frame_button = QPushButton('Reset frame')
        self.reset_frame_button.clicked.connect(controller.on_reset_mask)
        self.reset_object_button = QPushButton('Reset object')
        self.reset_object_button.clicked.connect(controller.on_reset_object)
        self.remove_object_all_button = QPushButton('Remove object (all frames)')
        self.remove_object_all_button.clicked.connect(controller.on_remove_object_all_frames)

        # Mask slot management
        self.snapshot_masks_button = QPushButton('Snapshot masks')
        self.snapshot_masks_button.setToolTip('Copy current masks to a new slot for comparison')
        self.snapshot_masks_button.clicked.connect(controller.on_snapshot_masks)

        self.switch_mask_slot_button = QPushButton('Switch [masks] (0)')
        self.switch_mask_slot_button.setToolTip('Cycle through mask snapshots')
        self.switch_mask_slot_button.clicked.connect(controller.on_switch_mask_slot)

        self.clear_masks_to_end_button = QPushButton('Clear masks → end')
        self.clear_masks_to_end_button.setToolTip('Delete all masks from current frame to the end')
        self.clear_masks_to_end_button.clicked.connect(controller.on_clear_masks_to_end)

        # set up the LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(150)
        self.lcd.setText('{: 5d} / {: 5d}'.format(0, controller.T - 1))

        # ID
        self.object_dial = QSpinBox()

        self.object_class_combo = QComboBox()
        for obj_id in range(1, controller.num_objects + 1):
            class_name = custom_names[obj_id]  # assuming `custom_names` is a list or dict
            self.object_class_combo.addItem(class_name, obj_id)  # store obj_id as userData

        self.object_class_combo.currentIndexChanged.connect(self.on_class_combo_changed)

        self.object_dial.setReadOnly(False)
        self.object_dial.setMinimumSize(50, 30)
        self.object_dial.setMinimum(1)
        self.object_dial.setMaximum(controller.num_objects)
        self.object_dial.editingFinished.connect(controller.on_object_dial_change)

        self.object_color = QLabel()
        self.object_color.setMinimumSize(30, 30)
        self.object_color.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.frame_name = QLabel()
        self.frame_name.setMinimumSize(100, 30)
        self.frame_name.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.soft_mask_indicator = QLabel('')
        self.soft_mask_indicator.setFixedHeight(20)
        self.soft_mask_indicator.setStyleSheet('color: #888; font-size: 10px;')

        # timeline slider
        self.tl_slider = MarkerSlider(Qt.Orientation.Horizontal)
        self.tl_slider.valueChanged.connect(controller.on_slider_update)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(controller.T - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tl_slider.setTickInterval(1)

        # combobox
        self.combo = QComboBox(self)
        self.combo.addItem("image")
        self.combo.addItem("mask")
        self.combo.addItem("davis")
        self.combo.addItem("fade")
        self.combo.addItem("light")
        self.combo.addItem("popup")
        self.combo.addItem("rgba")
        self.combo.addItem("soft")
        self.combo.setCurrentText('davis')
        self.combo.currentTextChanged.connect(controller.set_vis_mode)

        self.combo.setCurrentText('None')

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_mouse_press
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True)  # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release

        # clearing memory
        self.clear_all_mem_button = QPushButton('Reset all memory')
        self.clear_all_mem_button.clicked.connect(controller.on_clear_memory)
        self.clear_non_perm_mem_button = QPushButton('Reset non-permanent memory')
        self.clear_non_perm_mem_button.clicked.connect(controller.on_clear_non_permanent_memory)

        # displaying memory usage
        self.perm_mem_gauge, self.perm_mem_gauge_layout = create_gauge('Permanent memory size')
        self.work_mem_gauge, self.work_mem_gauge_layout = create_gauge('Working memory size')
        self.long_mem_gauge, self.long_mem_gauge_layout = create_gauge('Long-term memory size')
        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge(
            'GPU mem. (all proc, w/ caching)')
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge(
            'GPU mem. (torch, w/o caching)')

        # Parameters setting
        self.work_mem_min, self.work_mem_min_layout = create_parameter_box(
            1, 100, 'Min. working memory frames', callback=controller.on_work_min_change)
        self.work_mem_max, self.work_mem_max_layout = create_parameter_box(
            2, 100, 'Max. working memory frames', callback=controller.on_work_max_change)
        self.long_mem_max, self.long_mem_max_layout = create_parameter_box(
            1000,
            100000,
            'Max. long-term memory size',
            step=1000,
            callback=controller.update_config)
        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(
            1, 100, 'Memory frame every (r)', callback=controller.update_config)

        # import mask/layer
        self.import_mask_button = QPushButton('Import mask')
        self.import_mask_button.clicked.connect(controller.on_import_mask)
        self.import_layer_button = QPushButton('Import layer')
        self.import_layer_button.clicked.connect(controller.on_import_layer)

        # auto-segment (SurgNetXL)
        self.auto_seg_button = QPushButton('Auto-segment (A)')
        self.auto_seg_button.clicked.connect(controller.on_auto_segment)

        # change point detection (SurgeNetXL embeddings)
        self.detect_changes_button = QPushButton('Detect Changes (H)')
        self.detect_changes_button.clicked.connect(controller.on_detect_changes)
        self.change_sensitivity, self.change_sensitivity_layout = create_parameter_box(
            1, 200, 'Change sensitivity %', step=5,
            callback=controller.on_change_sensitivity_update)
        self.change_sensitivity.setValue(50)

        # Global memory (cross-video exemplar pairs)
        self.global_mem_list = QListWidget()
        self.global_mem_list.setMaximumHeight(100)
        self.global_mem_list.itemDoubleClicked.connect(controller.on_open_global_memory_folder)

        self.save_global_mem_button = QPushButton('Save (G)')
        self.save_global_mem_button.clicked.connect(controller.on_save_to_global_memory)
        self.save_all_global_mem_button = QPushButton('Save All')
        self.save_all_global_mem_button.clicked.connect(controller.on_save_all_to_global_memory)
        self.load_folder_button = QPushButton('Load Folder')
        self.load_folder_button.clicked.connect(controller.on_load_folder_from_global_memory)
        self.load_all_button = QPushButton('Load All')
        self.load_all_button.clicked.connect(controller.on_load_all_from_global_memory)
        self.remove_global_mem_button = QPushButton('Remove')
        self.remove_global_mem_button.clicked.connect(controller.on_remove_global_memory_folder)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # Open workspace folder
        self.open_workspace_button = QPushButton('Open Workspace Folder')
        self.open_workspace_button.clicked.connect(controller.on_open_workspace)

        # ── Class Power Panel (left side) ──
        self._build_class_power_panel(controller)

        # navigator
        navi = QHBoxLayout()

        interact_subbox = QVBoxLayout()
        interact_topbox = QHBoxLayout()
        interact_botbox = QHBoxLayout()
        interact_topbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        interact_topbox.addWidget(self.lcd)
        interact_topbox.addWidget(self.play_button)
        interact_topbox.addWidget(self.play_x4_button)
        interact_topbox.addWidget(self.reset_frame_button)
        interact_topbox.addWidget(self.reset_object_button)
        interact_topbox.addWidget(self.remove_object_all_button)
        interact_topbox.addWidget(self.snapshot_masks_button)
        interact_topbox.addWidget(self.switch_mask_slot_button)
        interact_topbox.addWidget(self.clear_masks_to_end_button)
        interact_topbox.addWidget(self.frame_name)
        interact_topbox.addWidget(self.soft_mask_indicator)

        interact_botbox.addWidget(self.object_color)
        interact_botbox.addWidget(QLabel('ID:'))
        interact_botbox.addWidget(self.object_dial)
        interact_botbox.addWidget(QLabel('Visualization mode'))
        interact_botbox.addWidget(self.combo)

        interact_subbox.addLayout(interact_topbox)
        interact_subbox.addLayout(interact_botbox)
        interact_botbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        navi.addLayout(interact_subbox)

        apply_fixed_size_policy = lambda x: x.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.
                                                            Policy.Fixed)
        apply_to_all_children_widget(interact_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(interact_botbox, apply_fixed_size_policy)

        navi.addStretch(1)
        navi.addStretch(1)
        overlay_subbox = QVBoxLayout()
        overlay_topbox = QHBoxLayout()
        overlay_botbox = QHBoxLayout()
        overlay_topbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        overlay_botbox.setAlignment(Qt.AlignmentFlag.AlignLeft)
        overlay_subbox.addLayout(overlay_topbox)
        overlay_subbox.addLayout(overlay_botbox)
        navi.addLayout(overlay_subbox)
        apply_to_all_children_widget(overlay_topbox, apply_fixed_size_policy)
        apply_to_all_children_widget(overlay_botbox, apply_fixed_size_policy)

        navi.addStretch(1)
        control_subbox = QVBoxLayout()
        control_topbox = QHBoxLayout()
        control_botbox = QHBoxLayout()
        control_topbox.addWidget(self.commit_button)
        control_topbox.addWidget(self.forward_run_button)
        control_topbox.addWidget(self.backward_run_button)
        control_topbox.addWidget(self.fill_gaps_checkbox)
        control_topbox.addWidget(self.crf_checkbox)
        control_topbox.addWidget(self.save_soft_mask_checkbox)
        control_botbox.addWidget(self.progressbar)
        control_subbox.addLayout(control_topbox)
        control_subbox.addLayout(control_botbox)
        navi.addLayout(control_subbox)

        # Drawing area main canvas
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.class_power_panel, 0)
        draw_area.addWidget(self.main_canvas, 4)

        # right area
        right_area = QVBoxLayout()
        right_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        right_area.addWidget(self.open_workspace_button)

        # Parameters
        right_area.addLayout(self.perm_mem_gauge_layout)
        right_area.addLayout(self.work_mem_gauge_layout)
        right_area.addLayout(self.long_mem_gauge_layout)
        right_area.addLayout(self.gpu_mem_gauge_layout)
        right_area.addLayout(self.torch_mem_gauge_layout)
        right_area.addWidget(self.clear_all_mem_button)
        right_area.addWidget(self.clear_non_perm_mem_button)
        right_area.addLayout(self.work_mem_min_layout)
        right_area.addLayout(self.work_mem_max_layout)
        right_area.addLayout(self.long_mem_max_layout)
        right_area.addLayout(self.mem_every_box_layout)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        import_area.addWidget(self.auto_seg_button)
        right_area.addLayout(import_area)
        right_area.addWidget(self.detect_changes_button)
        right_area.addLayout(self.change_sensitivity_layout)

        # Global memory
        right_area.addWidget(QLabel('Global Memory (cross-video)'))
        right_area.addWidget(self.global_mem_list)
        global_mem_buttons = QHBoxLayout()
        global_mem_buttons.addWidget(self.save_global_mem_button)
        global_mem_buttons.addWidget(self.save_all_global_mem_button)
        global_mem_buttons.addWidget(self.load_folder_button)
        global_mem_buttons.addWidget(self.load_all_button)
        global_mem_buttons.addWidget(self.remove_global_mem_button)
        right_area.addLayout(global_mem_buttons)

        # console
        right_area.addWidget(self.console)

        draw_area.addLayout(right_area, 1)

        layout = QVBoxLayout()
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        self.setLayout(layout)

        # timer to play video
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(controller.on_play_video_timer)

        # timer to play video at x4 speed
        self.timer_x4 = QTimer()
        self.timer_x4.setSingleShot(False)
        self.timer_x4.timeout.connect(controller.on_play_video_timer_x4)

        # timer to update GPU usage
        self.gpu_timer = QTimer()
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(controller.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # Objects shortcuts
        for i in range(1, controller.num_objects + 1):
            QShortcut(QKeySequence(str(i)),
                      self).activated.connect(functools.partial(controller.hit_number_key, i))
            QShortcut(QKeySequence(f"Ctrl+{i}"),
                      self).activated.connect(functools.partial(controller.hit_number_key, i))

        # next/prev frame shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(controller.on_prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(controller.on_next_frame)

        # +/- 10 frames shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 10))
        QShortcut(QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(functools.partial(controller.on_next_frame, 10))
        
        # first/last frame shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.AltModifier),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 999999))
        QShortcut(QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.AltModifier),
                    self).activated.connect(functools.partial(controller.on_next_frame, 999999))
        
        # jump to next/prev marker
        QShortcut(QKeySequence(Qt.Key.Key_Up), self).activated.connect(controller.on_next_marker)
        QShortcut(QKeySequence(Qt.Key.Key_Down), self).activated.connect(controller.on_prev_marker)

        # single-frame propagation
        QShortcut(QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.ControlModifier),
                    self).activated.connect(controller.on_propagate_forward_one)
        QShortcut(QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.ControlModifier),
                    self).activated.connect(controller.on_propagate_backward_one)

        # 10-frame propagation
        _ctrl_shift = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        QShortcut(QKeySequence(QKeyCombination(_ctrl_shift, Qt.Key.Key_Right)),
                    self).activated.connect(functools.partial(controller.on_propagate_forward_one, 10))
        QShortcut(QKeySequence(QKeyCombination(_ctrl_shift, Qt.Key.Key_Left)),
                    self).activated.connect(functools.partial(controller.on_propagate_backward_one, 10))

        # commit to permanent memory shortcut
        QShortcut(QKeySequence(Qt.Key.Key_C), self).activated.connect(controller.on_commit)

        # propagate forward/backward/pause shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_F), self).activated.connect(controller.on_forward_propagation)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(controller.on_forward_propagation)
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(controller.on_backward_propagation)
        
        # Toggle visualization mode
        QShortcut(QKeySequence(Qt.Key.Key_T), self).activated.connect(controller.on_toggle_vis_mode)

        # F1-F8: switch visualization mode directly
        vis_modes = ['image', 'mask', 'davis', 'fade', 'light', 'popup', 'rgba', 'soft']
        fkeys = [Qt.Key.Key_F1, Qt.Key.Key_F2, Qt.Key.Key_F3,
                 Qt.Key.Key_F4, Qt.Key.Key_F5, Qt.Key.Key_F6, Qt.Key.Key_F7, Qt.Key.Key_F8]
        for key, mode in zip(fkeys, vis_modes):
            QShortcut(QKeySequence(key), self).activated.connect(
                functools.partial(controller.set_vis_mode_direct, mode))

        # auto-segment current frame
        QShortcut(QKeySequence(Qt.Key.Key_A), self).activated.connect(controller.on_auto_segment)

        # change point detection
        QShortcut(QKeySequence(Qt.Key.Key_H), self).activated.connect(controller.on_detect_changes)
        QShortcut(QKeySequence(Qt.Key.Key_J), self).activated.connect(controller.on_next_change_marker)
        QShortcut(QKeySequence(Qt.Key.Key_K), self).activated.connect(controller.on_prev_change_marker)
        QShortcut(QKeySequence(Qt.Key.Key_H | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(controller.on_clear_change_markers)
        QShortcut(QKeySequence(Qt.Key.Key_V), self).activated.connect(
            controller.on_toggle_change_heatmap)
        QShortcut(QKeySequence(Qt.Key.Key_M), self).activated.connect(
            controller.on_toggle_mask_diff)

        # save to global memory
        QShortcut(QKeySequence(Qt.Key.Key_G),
                    self).activated.connect(controller.on_save_to_global_memory)

        # remove object from all frames
        QShortcut(QKeySequence(Qt.Key.Key_D | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(controller.on_remove_object_all_frames)

        # undo last mask edit
        QShortcut(QKeySequence(Qt.Key.Key_Z | Qt.KeyboardModifier.ControlModifier),
                    self).activated.connect(controller.on_undo)

        # CRF
        QShortcut(QKeySequence(Qt.Key.Key_R), self).activated.connect(controller.on_apply_crf_current_frame)

        # uncertainty marker navigation
        QShortcut(QKeySequence(Qt.Key.Key_N), self).activated.connect(controller.on_next_uncertainty_marker)
        QShortcut(QKeySequence(Qt.Key.Key_N | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(controller.on_clear_uncertainty_markers)

        # brush / eraser mode
        QShortcut(QKeySequence(Qt.Key.Key_P), self).activated.connect(controller.on_toggle_brush_mode)
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft),
                    self).activated.connect(functools.partial(controller.on_brush_size_change, -2))
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight),
                    self).activated.connect(functools.partial(controller.on_brush_size_change, 2))

        # mask slot management
        QShortcut(QKeySequence(Qt.Key.Key_S | Qt.KeyboardModifier.ControlModifier),
                    self).activated.connect(controller.on_snapshot_masks)
        QShortcut(QKeySequence(Qt.Key.Key_W),
                    self).activated.connect(controller.on_switch_mask_slot)
        QShortcut(QKeySequence(Qt.Key.Key_X | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(controller.on_clear_masks_to_end)

        # quit shortcut
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.close)

    def set_current_object_id(self, object_id: int):
        self.object_dial.blockSignals(True)
        self.object_dial.setValue(object_id)
        self.object_dial.blockSignals(False)

        # Update combo box to match object_id
        index = self.object_class_combo.findData(object_id)
        if index != -1:
            self.object_class_combo.blockSignals(True)
            self.object_class_combo.setCurrentIndex(index)
            self.object_class_combo.blockSignals(False)

        self.set_object_color(object_id)

    def on_class_combo_changed(self, index):
        obj_id = self.object_class_combo.itemData(index)
        if obj_id is None:
            return

        self.object_dial.blockSignals(True)
        self.object_dial.setValue(obj_id)
        self.object_dial.blockSignals(False)

        self.controller.on_object_dial_change()

    def resizeEvent(self, event):
        self.controller.show_current_frame()

    def text(self, text):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text + '\n')

    def set_canvas(self, image):
        height, width, channel = image.shape
        # if the image is RGBA, convert to RGB first by coloring the background green
        if channel == 4:
            image_rgb = image[:, :, :3].copy()
            alpha = image[:, :, 3].astype(np.float32) / 255
            green_bg = np.array([0, 255, 0])
            # soft blending
            image = (image_rgb * alpha[:, :, np.newaxis] + green_bg[np.newaxis, np.newaxis, :] *
                     (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

        bytesPerLine = 3 * width

        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(self.main_canvas.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation)))

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_slider(self, value):
        self.lcd.setText('{: 3d} / {: 3d}'.format(value, self.controller.T - 1))
        self.tl_slider.setValue(value)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        out_of_bound = ((x < 0) or (y < 0) or (x > self.w - 1) or (y > self.h - 1))

        return out_of_bound

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))

        return x, y

    def forward_propagation_start(self):
        self.backward_run_button.setEnabled(False)
        self.forward_run_button.setText('Pause propagation')

    def backward_propagation_start(self):
        self.forward_run_button.setEnabled(False)
        self.backward_run_button.setText('Pause propagation')

    def pause_propagation(self):
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_all_mem_button.setEnabled(True)
        self.clear_non_perm_mem_button.setEnabled(True)
        self.forward_run_button.setText('Propagate forward')
        self.backward_run_button.setText('propagate backward')
        self.tl_slider.setEnabled(True)

    def process_events(self):
        QApplication.processEvents()

    def on_mouse_press(self, event):
        event.accept()
        if self.is_pos_out_of_bound(event.position().x(), event.position().y()):
            return

        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                action = 'pick'
            else:
                action = 'left'
        elif event.button() == Qt.MouseButton.RightButton:
            action = 'right'
        elif event.button() == Qt.MouseButton.MiddleButton:
            action = 'middle'
        else:
            return

        self.click_fn(action, ex, ey)

    def on_mouse_motion(self, event):
        event.accept()
        ex, ey = self.get_scaled_pos(event.position().x(), event.position().y())
        self._mouse_buttons = event.buttons()
        self.on_mouse_motion_xy(ex, ey)

    def on_mouse_release(self, event):
        event.accept()
        self.on_mouse_release_fn()

    def on_play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play video')
        else:
            # stop x4 if running
            if self.timer_x4.isActive():
                self.timer_x4.stop()
                self.play_x4_button.setText('Play x4')
            self.timer.start(1000 // 30)
            self.play_button.setText('Stop video')

    def on_play_video_x4(self):
        if self.timer_x4.isActive():
            self.timer_x4.stop()
            self.play_x4_button.setText('Play x4')
        else:
            # stop normal play if running
            if self.timer.isActive():
                self.timer.stop()
                self.play_button.setText('Play video')
            self.timer_x4.start(1000 // 30)
            self.play_x4_button.setText('Stop x4')

    def open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   prompt,
                                                   "",
                                                   "Image files (*)",
                                                   options=options)
        return file_name

    def set_object_color(self, object_id: int):
        r, g, b = custom_palette_np[object_id]
        rgb = f'rgb({r},{g},{b})'
        self.object_color.setFixedSize(50, 30)  # Make it square
        self.object_color.setStyleSheet(f'QLabel {{ background-color: {rgb}; border: 1px solid #d3d3d3; }}')

    # ── Class Power Panel helpers ──────────────────────────────────────

    def _build_class_power_panel(self, controller):
        """Create a scrollable left panel with one row per class: color swatch,
        name button (click to select), and a power slider."""
        num_objects = controller.num_objects

        self.class_power_panel = QFrame()
        self.class_power_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.class_power_panel.setFixedWidth(320)

        panel_layout = QVBoxLayout(self.class_power_panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        panel_layout.setSpacing(2)

        # Title
        title = QLabel('Class Power')
        title.setStyleSheet('font-weight: bold;')
        panel_layout.addWidget(title)

        # Mode toggle: Multiply / Exponent
        mode_row = QHBoxLayout()
        self.power_mode_group = QButtonGroup(self)
        self.power_mode_multiply = QRadioButton('Multiply')
        self.power_mode_exponent = QRadioButton('Exponent')
        self.power_mode_multiply.setChecked(True)
        self.power_mode_group.addButton(self.power_mode_multiply)
        self.power_mode_group.addButton(self.power_mode_exponent)
        mode_row.addWidget(self.power_mode_multiply)
        mode_row.addWidget(self.power_mode_exponent)
        panel_layout.addLayout(mode_row)
        self.power_mode_group.buttonClicked.connect(
            lambda _btn: controller.on_class_power_mode_changed())

        # Global soft mask power slider
        global_label_row = QHBoxLayout()
        global_label_row.setSpacing(4)
        global_lbl = QLabel('Global')
        global_lbl.setStyleSheet('font-weight: bold;')
        global_label_row.addWidget(global_lbl)
        self.global_power_label = QLabel('1.00')
        self.global_power_label.setFixedWidth(40)
        self.global_power_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        global_label_row.addWidget(self.global_power_label)
        panel_layout.addLayout(global_label_row)

        self.global_power_slider = QSlider(Qt.Orientation.Horizontal)
        self.global_power_slider.setMinimum(1)
        self.global_power_slider.setMaximum(10000)
        self.global_power_slider.setValue(100)
        self.global_power_slider.setMinimumHeight(20)
        self.global_power_slider.setToolTip('Global soft mask multiplier (1.0 = neutral)')
        self.global_power_slider.valueChanged.connect(self._on_global_power_slider_changed)
        panel_layout.addWidget(self.global_power_slider)

        # Reset all button
        self.power_reset_button = QPushButton('Reset All to 1.0')
        self.power_reset_button.clicked.connect(controller.on_class_power_reset)
        panel_layout.addWidget(self.power_reset_button)

        # Scrollable area for class rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        self._class_rows_layout = QVBoxLayout(scroll_widget)
        self._class_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._class_rows_layout.setSpacing(2)

        self._class_power_sliders: list[QSlider] = []
        self._class_power_labels: list[QLabel] = []
        self._class_name_buttons: list[QPushButton] = []

        for obj_id in range(1, num_objects + 1):
            # Top row: color swatch + class name button + value label
            top_row = QHBoxLayout()
            top_row.setSpacing(4)

            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            r, g, b = custom_palette_np[obj_id]
            swatch.setStyleSheet(
                f'background-color: rgb({r},{g},{b}); border: 1px solid #888;')
            top_row.addWidget(swatch)

            name = custom_names.get(obj_id, f'Class {obj_id}')
            btn = QPushButton(name)
            btn.setFixedHeight(24)
            btn.setToolTip(f'Select class {obj_id}: {name}')
            btn.setStyleSheet('text-align: left; padding: 1px 4px; font-size: 10px;')
            btn.clicked.connect(functools.partial(controller.hit_number_key, obj_id))
            self._class_name_buttons.append(btn)
            top_row.addWidget(btn, 1)

            val_label = QLabel('1.00')
            val_label.setFixedWidth(40)
            val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._class_power_labels.append(val_label)
            top_row.addWidget(val_label)

            # Bottom row: slider spanning full width
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(1)
            slider.setMaximum(10000)
            slider.setValue(100)
            slider.setMinimumHeight(18)
            slider.setToolTip('Class power weight (1.0 = neutral)')
            slider.valueChanged.connect(
                functools.partial(self._on_class_power_slider_changed, obj_id))
            self._class_power_sliders.append(slider)

            self._class_rows_layout.addLayout(top_row)
            self._class_rows_layout.addWidget(slider)

        self._class_rows_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        panel_layout.addWidget(scroll, 1)

    def _on_class_power_slider_changed(self, obj_id: int, value: int):
        """Called when any class power slider moves."""
        idx = obj_id - 1  # 0-based index into our lists
        real_value = value / 100.0
        self._class_power_labels[idx].setText(f'{real_value:.2f}')
        self.controller.on_class_power_changed(obj_id, real_value)

    def _on_global_power_slider_changed(self, value: int):
        """Called when the global soft mask power slider moves."""
        real_value = value / 100.0
        self.global_power_label.setText(f'{real_value:.2f}')
        self.controller.on_global_power_changed(real_value)

    def get_class_power_mode(self) -> str:
        """Return 'multiply' or 'exponent'."""
        if self.power_mode_exponent.isChecked():
            return 'exponent'
        return 'multiply'

    def get_all_class_power_weights(self) -> list[float]:
        """Return list of power weights for objects 1..N (background always 1.0)."""
        return [s.value() / 100.0 for s in self._class_power_sliders]

    def reset_class_power_sliders(self):
        """Reset all sliders (including global) to 1.0."""
        self.global_power_slider.blockSignals(True)
        self.global_power_slider.setValue(100)
        self.global_power_slider.blockSignals(False)
        self.global_power_label.setText('1.00')
        for slider in self._class_power_sliders:
            slider.blockSignals(True)
            slider.setValue(100)
            slider.blockSignals(False)
        for label in self._class_power_labels:
            label.setText('1.00')

    def highlight_selected_class(self, obj_id: int):
        """Visually highlight the selected class row."""
        for i, btn in enumerate(self._class_name_buttons):
            if i + 1 == obj_id:
                btn.setStyleSheet(
                    'text-align: left; padding: 1px 4px; '
                    'font-weight: bold; border: 2px solid #4488ff;')
            else:
                btn.setStyleSheet('text-align: left; padding: 1px 4px;')

    def update_global_memory_list(self, folders, current_video=''):
        """Refresh the global memory list with ``[(name, count), ...]``."""
        self.global_mem_list.clear()
        for name, count in folders:
            item = QListWidgetItem(f'{name}  ({count} files)')
            if name == current_video:
                item.setForeground(QColor(0, 200, 0))
            self.global_mem_list.addItem(item)

    def progressbar_update(self, progress: float):
        self.progressbar.setValue(int(progress * 100))
        self.process_events()
