"""QOpenGLWidget-based canvas for zero-copy GPU rendering.

Replaces the QLabel canvas with an OpenGL surface that composites
image + mask/prob overlays entirely on the GPU via GLSL shaders.

Two upload paths:
  - CPU (numpy)  : glTexSubImage2D   – used during scrubbing
  - GPU (tensor) : CUDA-GL interop   – used during propagation
"""

import ctypes
import logging
from typing import List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat

try:
    from OpenGL.GL import *  # noqa
    from OpenGL.GL import shaders as gl_shaders_mod
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from gui.gl_shaders import (
    VERTEX_SHADER, FRAGMENT_SHADER,
    MODE_FROM_NAME, MODE_PRECOMPOSITED,
)

log = logging.getLogger(__name__)


def _check_opengl():
    if not HAS_OPENGL:
        raise ImportError("PyOpenGL is not installed. Install with: pip install PyOpenGL")


class GLCanvasWidget(QOpenGLWidget):
    """OpenGL canvas that composites image + mask overlays via GLSL."""

    def __init__(self, image_h: int, image_w: int, num_objects: int,
                 color_map_np: np.ndarray, parent=None):
        _check_opengl()

        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        self.image_h = image_h
        self.image_w = image_w
        self.num_objects = num_objects
        self.num_classes = num_objects + 1  # including background

        # Color map: (256, 3) uint8 → will be uploaded as float32
        self._color_map_np = color_map_np.copy()

        # GL resource handles (set in initializeGL)
        self._program = None
        self._vao = None
        self._vbo = None
        self._tex_image = None
        self._tex_mask = None
        self._tex_color_map = None
        self._tex_overlay = None
        self._tex_prob = None

        # Uniform locations
        self._u = {}

        # Display rect for aspect-ratio viewport
        self._display_rect = (0, 0, 1, 1)

        # For coordinate mapping compatibility
        self.main_canvas_size = QSize(1, 1)
        self.image_size = QSize(image_w, image_h)

        # CUDA interop (initialized later in Phase 2)
        self._interop = None
        self._interop_available = False
        self._interop_resources = {}

        # Current state
        self._mode = MODE_FROM_NAME['davis']
        self._alpha = 0.5
        self._use_soft_prob = False
        self._target_objects = []
        self._gl_initialized = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(100, 100)
        self.setMouseTracking(True)

    # ── QOpenGLWidget lifecycle ───────────────────────────────────────────

    def initializeGL(self):
        glClearColor(0.12, 0.12, 0.12, 1.0)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # RGB rows (3 bytes/pixel) are not always 4-byte aligned — tell OpenGL
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        self._compile_shaders()
        self._create_fullscreen_quad()
        self._create_textures()
        self._upload_color_map()

        # Try CUDA-GL interop (Phase 2 will populate _init_cuda_interop)
        try:
            self._init_cuda_interop()
        except Exception as e:
            log.info("CUDA-GL interop not available: %s", e)
            self._interop_available = False

        self._gl_initialized = True
        log.info("GLCanvasWidget initialized: %dx%d, %d objects, interop=%s",
                 self.image_w, self.image_h, self.num_objects, self._interop_available)

    def resizeGL(self, w: int, h: int):
        img_aspect = self.image_w / self.image_h
        wnd_aspect = w / h

        if wnd_aspect > img_aspect:
            # pillarbox
            disp_h = h
            disp_w = int(h * img_aspect)
        else:
            # letterbox
            disp_w = w
            disp_h = int(w / img_aspect)

        disp_x = (w - disp_w) // 2
        disp_y = (h - disp_h) // 2
        self._display_rect = (disp_x, disp_y, disp_w, disp_h)
        self.main_canvas_size = QSize(w, h)

    def paintGL(self):
        if self._program is None:
            return

        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._program)

        # Bind textures to units
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_image)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._tex_mask)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_1D, self._tex_color_map)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, self._tex_overlay)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_prob)

        # Set uniforms
        glUniform1i(self._u['tex_image'], 0)
        glUniform1i(self._u['tex_mask'], 1)
        glUniform1i(self._u['tex_color_map'], 2)
        glUniform1i(self._u['tex_overlay'], 3)
        glUniform1i(self._u['tex_prob'], 4)

        glUniform1i(self._u['u_mode'], self._mode)
        glUniform1f(self._u['u_alpha'], self._alpha)
        glUniform1i(self._u['u_num_classes'], self.num_classes)
        glUniform1i(self._u['u_use_soft_prob'], int(self._use_soft_prob))
        glUniform1i(self._u['u_num_targets'], len(self._target_objects))

        # Target objects array
        targets = self._target_objects[:256]
        padded = targets + [0] * (256 - len(targets))
        glUniform1iv(self._u['u_target_objects'], 256, padded)

        # Viewport to aspect-correct region
        x, y, w, h = self._display_rect
        glViewport(x, y, w, h)

        # Draw fullscreen quad
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glUseProgram(0)

    # ── Shader compilation ────────────────────────────────────────────────

    def _compile_shaders(self):
        vs = gl_shaders_mod.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fs = gl_shaders_mod.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self._program = gl_shaders_mod.compileProgram(vs, fs)

        # Cache uniform locations
        uniforms = [
            'tex_image', 'tex_mask', 'tex_color_map', 'tex_overlay', 'tex_prob',
            'u_mode', 'u_alpha', 'u_num_classes', 'u_use_soft_prob',
            'u_target_objects', 'u_num_targets',
        ]
        for name in uniforms:
            self._u[name] = glGetUniformLocation(self._program, name)

    # ── Fullscreen quad ───────────────────────────────────────────────────

    def _create_fullscreen_quad(self):
        # pos (x,y) + texcoord (u,v) — Y-flipped for image convention
        vertices = np.array([
            -1, -1, 0, 1,   # bottom-left
             1, -1, 1, 1,   # bottom-right
            -1,  1, 0, 0,   # top-left
             1,  1, 1, 0,   # top-right
        ], dtype=np.float32)

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # a_position (location 0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # a_texcoord (location 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    # ── Texture creation ──────────────────────────────────────────────────

    def _create_textures(self):
        W, H = self.image_w, self.image_h
        K = self.num_classes

        # Image texture: RGB float32
        self._tex_image = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_image)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, W, H, 0,
                     GL_RGB, GL_FLOAT, None)
        self._set_tex_params_2d()

        # Hard mask texture: R8 (class index 0-255) — NEAREST to avoid interpolation
        self._tex_mask = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_mask)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W, H, 0,
                     GL_RED, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Color map: 1D texture, 256 RGB float entries
        self._tex_color_map = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self._tex_color_map)
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, 256, 0,
                     GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)

        # Overlay texture: RGBA float32
        self._tex_overlay = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_overlay)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0,
                     GL_RGBA, GL_FLOAT, None)
        self._set_tex_params_2d()

        # Soft probability texture array: K layers of R32F
        self._tex_prob = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_prob)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, W, H, K, 0,
                     GL_RED, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def _set_tex_params_2d(self):
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def _upload_color_map(self):
        # Convert (256, 3) uint8 → (256, 3) float32 [0, 1]
        cmap = self._color_map_np.astype(np.float32) / 255.0
        cmap = np.ascontiguousarray(cmap)
        glBindTexture(GL_TEXTURE_1D, self._tex_color_map)
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGB, GL_FLOAT, cmap)

    # ── CUDA-GL interop (Phase 2 stub) ────────────────────────────────────

    def _init_cuda_interop(self):
        """Initialize CUDA-GL interop: register textures with CUDA."""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        from gui.cuda_gl_interop import CudaGLInterop
        from OpenGL.GL import GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY

        self._interop = CudaGLInterop()
        self._interop_resources['image'] = self._interop.register_texture(
            self._tex_image, GL_TEXTURE_2D)
        self._interop_resources['prob'] = self._interop.register_texture(
            self._tex_prob, GL_TEXTURE_2D_ARRAY)
        self._interop_resources['overlay'] = self._interop.register_texture(
            self._tex_overlay, GL_TEXTURE_2D)
        self._interop_available = True

    # ── Public API: GPU fast path (tensors) ───────────────────────────────

    def update_from_gpu(self, image_tensor, prob_tensor,
                        mode: str, target_objects: List[int],
                        alpha: float = 0.5,
                        overlay_tensor=None):
        """Upload GPU tensors via CUDA-GL interop and schedule repaint.

        Args:
            image_tensor: (3, H, W) float32 CUDA tensor [0, 1]
            prob_tensor:  (K+1, H, W) float32 CUDA tensor [0, 1]
            mode:         visualization mode name
            target_objects: list of object indices considered foreground
            alpha:        blend alpha for davis/light modes
            overlay_tensor: optional (4, H, W) float32 CUDA tensor
        """
        if not self._gl_initialized:
            return

        self.makeCurrent()

        if self._interop_available:
            self._cuda_upload_image(image_tensor)
            self._cuda_upload_prob(prob_tensor)
            if overlay_tensor is not None:
                self._cuda_upload_overlay(overlay_tensor)
            self._use_soft_prob = True
        else:
            # Fallback: transfer to CPU and use numpy path
            import torch
            img_np = (image_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
            prob_np = prob_tensor.cpu().numpy()
            self._cpu_upload_image_np(img_np)
            self._cpu_upload_prob_np(prob_np)
            if overlay_tensor is not None:
                ov_np = (overlay_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
                self._cpu_upload_overlay_np(ov_np)
            self._use_soft_prob = True

        self._mode = MODE_FROM_NAME.get(mode, MODE_FROM_NAME['davis'])
        self._alpha = alpha
        self._target_objects = list(target_objects)

        self.doneCurrent()
        self.update()

    # ── Public API: numpy slow path (scrubbing) ───────────────────────────

    def update_from_numpy(self, image_np: np.ndarray, mask_np: np.ndarray,
                          mode: str, target_objects: List[int],
                          alpha: float = 0.5,
                          overlay_np: Optional[np.ndarray] = None):
        """Upload numpy arrays and schedule repaint.

        Args:
            image_np: (H, W, 3) uint8 RGB image
            mask_np:  (H, W) uint8 class index mask
            mode:     visualization mode name
            target_objects: list of target object indices
            alpha:    blend alpha
            overlay_np: optional (H, W, 4) uint8 RGBA overlay
        """
        if not self._gl_initialized:
            return

        self.makeCurrent()

        self._cpu_upload_image_np(image_np)
        self._cpu_upload_mask_np(mask_np)
        if overlay_np is not None:
            self._cpu_upload_overlay_np(overlay_np)

        self._use_soft_prob = False
        self._mode = MODE_FROM_NAME.get(mode, MODE_FROM_NAME['davis'])
        self._alpha = alpha
        self._target_objects = list(target_objects)

        self.doneCurrent()
        self.update()

    # ── Public API: legacy pre-composited image ───────────────────────────

    def set_canvas(self, image: np.ndarray):
        """Accept a pre-composited (H, W, 3) or (H, W, 4) uint8 numpy image.

        Used for brush cursor, polygon overlay, heatmap overlays — anything
        already composited on CPU.
        """
        if not self._gl_initialized:
            return

        self.makeCurrent()

        if image.ndim == 3 and image.shape[2] == 4:
            # RGBA → composite on green background (match existing behavior)
            rgb = image[:, :, :3].astype(np.float32)
            a = image[:, :, 3:4].astype(np.float32) / 255.0
            green = np.array([[[0, 255, 0]]], dtype=np.float32)
            composited = (rgb * a + green * (1.0 - a)).astype(np.uint8)
            self._cpu_upload_image_np(composited)
        else:
            self._cpu_upload_image_np(image)

        self._mode = MODE_PRECOMPOSITED
        self._use_soft_prob = False

        self.doneCurrent()
        self.update()

    # ── CPU upload helpers ────────────────────────────────────────────────

    def _cpu_upload_image_np(self, image: np.ndarray):
        """Upload (H, W, 3) uint8 or float32 image to tex_image."""
        image = np.ascontiguousarray(image)
        glBindTexture(GL_TEXTURE_2D, self._tex_image)
        if image.dtype == np.uint8:
            # Convert to float in-place on GPU via GL
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            self.image_w, self.image_h,
                            GL_RGB, GL_UNSIGNED_BYTE, image)
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            self.image_w, self.image_h,
                            GL_RGB, GL_FLOAT, image)

    def _cpu_upload_mask_np(self, mask: np.ndarray):
        """Upload (H, W) uint8 hard mask to tex_mask."""
        mask = np.ascontiguousarray(mask)
        glBindTexture(GL_TEXTURE_2D, self._tex_mask)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        self.image_w, self.image_h,
                        GL_RED, GL_UNSIGNED_BYTE, mask)

    def _cpu_upload_prob_np(self, prob: np.ndarray):
        """Upload (K+1, H, W) float32 probability array to tex_prob."""
        K = min(prob.shape[0], self.num_classes)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._tex_prob)
        for k in range(K):
            layer = np.ascontiguousarray(prob[k])
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0,
                            0, 0, k,
                            self.image_w, self.image_h, 1,
                            GL_RED, GL_FLOAT, layer)

    def _cpu_upload_overlay_np(self, overlay: np.ndarray):
        """Upload (H, W, 4) uint8 or float32 RGBA overlay to tex_overlay."""
        overlay = np.ascontiguousarray(overlay)
        glBindTexture(GL_TEXTURE_2D, self._tex_overlay)
        if overlay.dtype == np.uint8:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            self.image_w, self.image_h,
                            GL_RGBA, GL_UNSIGNED_BYTE, overlay)
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            self.image_w, self.image_h,
                            GL_RGBA, GL_FLOAT, overlay)

    # ── CUDA upload methods ─────────────────────────────────────────────

    def _cuda_upload_image(self, tensor):
        """Upload (3, H, W) float32 CUDA tensor to image texture."""
        # Permute to (H, W, 3) contiguous for the memcpy
        t = tensor.permute(1, 2, 0).contiguous()
        self._interop.copy_tensor_to_texture_2d(
            t, self._interop_resources['image'],
            self.image_w, self.image_h, 3)

    def _cuda_upload_prob(self, tensor):
        """Upload (K+1, H, W) float32 CUDA tensor to prob texture array."""
        t = tensor.contiguous()
        K = min(t.shape[0], self.num_classes)
        self._interop.copy_tensor_to_texture_array(
            t, self._interop_resources['prob'],
            self.image_w, self.image_h, K)

    def _cuda_upload_overlay(self, tensor):
        """Upload (4, H, W) float32 CUDA tensor to overlay texture."""
        t = tensor.permute(1, 2, 0).contiguous()
        self._interop.copy_tensor_to_texture_2d(
            t, self._interop_resources['overlay'],
            self.image_w, self.image_h, 4)

    # ── Mouse coordinate mapping ──────────────────────────────────────────

    def pixel_pos_to_image_pos(self, x: float, y: float) -> Tuple[float, float]:
        """Convert widget pixel coordinates to image coordinates."""
        rx, ry, rw, rh = self._display_rect
        if rw <= 0 or rh <= 0:
            return x, y
        ix = (x - rx) / rw * self.image_w
        iy = (y - ry) / rh * self.image_h
        return ix, iy

    def is_pos_out_of_bound(self, x: float, y: float) -> bool:
        ix, iy = self.pixel_pos_to_image_pos(x, y)
        return ix < 0 or iy < 0 or ix > self.image_w - 1 or iy > self.image_h - 1

    def get_scaled_pos(self, x: float, y: float) -> Tuple[float, float]:
        ix, iy = self.pixel_pos_to_image_pos(x, y)
        ix = max(0, min(self.image_w - 1, ix))
        iy = max(0, min(self.image_h - 1, iy))
        return ix, iy
