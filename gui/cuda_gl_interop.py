"""CUDA-OpenGL interop for zero-copy GPU texture uploads.

Registers OpenGL textures with CUDA so that PyTorch tensors can be
copied directly to GL textures via GPU-internal memcpy (~0.2ms for 1080p),
avoiding the CPU roundtrip entirely.

Two backends:
  1. PyCUDA  (preferred) – pycuda.gl.RegisteredImage
  2. ctypes  (fallback)  – direct calls to libcudart
"""

import ctypes
import logging
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

# Try to import PyCUDA
_PYCUDA_AVAILABLE = False
try:
    import pycuda.driver as cuda
    import pycuda.gl as cuda_gl
    _PYCUDA_AVAILABLE = True
except ImportError:
    pass

# Try to load libcudart for ctypes fallback
_CUDART = None
try:
    _CUDART = ctypes.CDLL('libcudart.so')
except OSError:
    try:
        _CUDART = ctypes.CDLL('libcudart.dylib')
    except OSError:
        pass


# ── ctypes CUDA definitions ──────────────────────────────────────────────

class _CudaArray(ctypes.Structure):
    pass

_CudaArray_p = ctypes.POINTER(_CudaArray)

# cudaGraphicsResource_t is an opaque pointer
_CudaGraphicsResource_p = ctypes.c_void_p

# cudaMemcpy kinds
_cudaMemcpyDeviceToDevice = 3

# cudaGraphicsRegisterFlags
_cudaGraphicsRegisterFlagsWriteDiscard = 2


class CudaGLInterop:
    """Manages CUDA-GL interop for a set of OpenGL textures."""

    def __init__(self):
        self._backend = None
        self._pycuda_context = None
        self._resources = {}  # gl_tex_id → resource handle

        if _PYCUDA_AVAILABLE:
            try:
                self._init_pycuda()
                self._backend = 'pycuda'
                log.info("CUDA-GL interop initialized with PyCUDA backend")
                return
            except Exception as e:
                log.warning("PyCUDA init failed: %s, trying ctypes", e)

        if _CUDART is not None:
            try:
                self._init_ctypes()
                self._backend = 'ctypes'
                log.info("CUDA-GL interop initialized with ctypes backend")
                return
            except Exception as e:
                log.warning("ctypes CUDA init failed: %s", e)

        raise RuntimeError("No CUDA-GL interop backend available. "
                           "Install pycuda or ensure libcudart is accessible.")

    # ── PyCUDA backend ────────────────────────────────────────────────────

    def _init_pycuda(self):
        # PyCUDA needs an explicit context; reuse the one PyTorch created
        cuda.init()
        # Get the current CUDA context (created by PyTorch)
        self._pycuda_context = cuda.Context.attach()

    def register_texture(self, gl_tex_id: int, gl_target: int) -> object:
        """Register an OpenGL texture for CUDA access.

        Args:
            gl_tex_id: OpenGL texture name (GLuint)
            gl_target: GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY, etc.

        Returns:
            An opaque resource handle.
        """
        if self._backend == 'pycuda':
            return self._pycuda_register(gl_tex_id, gl_target)
        else:
            return self._ctypes_register(gl_tex_id, gl_target)

    def copy_tensor_to_texture_2d(self, tensor: torch.Tensor,
                                   resource: object,
                                   width: int, height: int,
                                   channels: int):
        """Copy a (H, W, C) contiguous CUDA float32 tensor to a 2D texture.

        Args:
            tensor: contiguous (H, W, C) float32 CUDA tensor
            resource: handle from register_texture()
            width, height: texture dimensions
            channels: number of channels (3 for RGB, 4 for RGBA)
        """
        assert tensor.is_contiguous() and tensor.is_cuda
        if self._backend == 'pycuda':
            self._pycuda_copy_2d(tensor, resource, width, height, channels)
        else:
            self._ctypes_copy_2d(tensor, resource, width, height, channels)

    def copy_tensor_to_texture_array(self, tensor: torch.Tensor,
                                      resource: object,
                                      width: int, height: int,
                                      num_layers: int):
        """Copy a (K, H, W) contiguous CUDA float32 tensor to a 2D array texture.

        Each layer k of the tensor is copied to layer k of the texture array.

        Args:
            tensor: contiguous (K, H, W) float32 CUDA tensor
            resource: handle from register_texture()
            width, height: per-layer dimensions
            num_layers: number of layers to copy
        """
        assert tensor.is_contiguous() and tensor.is_cuda
        if self._backend == 'pycuda':
            self._pycuda_copy_array(tensor, resource, width, height, num_layers)
        else:
            self._ctypes_copy_array(tensor, resource, width, height, num_layers)

    def unregister(self, resource: object):
        """Unregister a previously registered texture."""
        if self._backend == 'pycuda':
            resource.unregister()
        elif self._backend == 'ctypes':
            _CUDART.cudaGraphicsUnregisterResource(resource)

    # ── PyCUDA implementation ─────────────────────────────────────────────

    def _pycuda_register(self, gl_tex_id, gl_target):
        return cuda_gl.RegisteredImage(
            int(gl_tex_id), gl_target,
            cuda_gl.graphics_map_flags.WRITE_DISCARD
        )

    def _pycuda_copy_2d(self, tensor, resource, width, height, channels):
        mapping = resource.map()
        try:
            arr = mapping.array(0, 0)
            cpy = cuda.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(arr)
            cpy.width_in_bytes = width * channels * 4  # float32
            cpy.src_pitch = cpy.width_in_bytes
            cpy.dst_pitch = cpy.width_in_bytes
            cpy.height = height
            cpy(aligned=True)
        finally:
            mapping.unmap()

    def _pycuda_copy_array(self, tensor, resource, width, height, num_layers):
        mapping = resource.map()
        try:
            for k in range(num_layers):
                arr = mapping.array(0, k)
                cpy = cuda.Memcpy2D()
                offset = k * height * width * 4  # float32, 1 channel per layer
                cpy.set_src_device(tensor.data_ptr() + offset)
                cpy.set_dst_array(arr)
                cpy.width_in_bytes = width * 4  # 1 channel R32F
                cpy.src_pitch = cpy.width_in_bytes
                cpy.dst_pitch = cpy.width_in_bytes
                cpy.height = height
                cpy(aligned=True)
        finally:
            mapping.unmap()

    # ── ctypes implementation ─────────────────────────────────────────────

    def _init_ctypes(self):
        # Verify cudart is functional
        ret = _CUDART.cudaGetDeviceCount(ctypes.byref(ctypes.c_int(0)))
        if ret != 0:
            raise RuntimeError(f"cudaGetDeviceCount failed with error {ret}")

    def _ctypes_register(self, gl_tex_id, gl_target):
        resource = ctypes.c_void_p()
        ret = _CUDART.cudaGraphicsGLRegisterImage(
            ctypes.byref(resource),
            ctypes.c_uint(int(gl_tex_id)),
            ctypes.c_uint(int(gl_target)),
            ctypes.c_uint(_cudaGraphicsRegisterFlagsWriteDiscard)
        )
        if ret != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterImage failed: {ret}")
        return resource

    def _ctypes_copy_2d(self, tensor, resource, width, height, channels):
        ret = _CUDART.cudaGraphicsMapResources(
            1, ctypes.byref(resource), ctypes.c_void_p(0))
        if ret != 0:
            raise RuntimeError(f"cudaGraphicsMapResources failed: {ret}")
        try:
            cuda_array = ctypes.c_void_p()
            ret = _CUDART.cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(cuda_array), resource, 0, 0)
            if ret != 0:
                raise RuntimeError(f"GetMappedArray failed: {ret}")

            ret = _CUDART.cudaMemcpy2DToArray(
                cuda_array,                        # dst array
                ctypes.c_size_t(0),                # dst x offset
                ctypes.c_size_t(0),                # dst y offset
                ctypes.c_void_p(tensor.data_ptr()), # src
                ctypes.c_size_t(width * channels * 4),  # src pitch
                ctypes.c_size_t(width * channels * 4),  # width in bytes
                ctypes.c_size_t(height),           # height
                ctypes.c_int(_cudaMemcpyDeviceToDevice)
            )
            if ret != 0:
                raise RuntimeError(f"cudaMemcpy2DToArray failed: {ret}")
        finally:
            _CUDART.cudaGraphicsUnmapResources(
                1, ctypes.byref(resource), ctypes.c_void_p(0))

    def _ctypes_copy_array(self, tensor, resource, width, height, num_layers):
        ret = _CUDART.cudaGraphicsMapResources(
            1, ctypes.byref(resource), ctypes.c_void_p(0))
        if ret != 0:
            raise RuntimeError(f"cudaGraphicsMapResources failed: {ret}")
        try:
            for k in range(num_layers):
                cuda_array = ctypes.c_void_p()
                ret = _CUDART.cudaGraphicsSubResourceGetMappedArray(
                    ctypes.byref(cuda_array), resource, k, 0)
                if ret != 0:
                    raise RuntimeError(f"GetMappedArray layer {k} failed: {ret}")

                offset = k * height * width * 4
                ret = _CUDART.cudaMemcpy2DToArray(
                    cuda_array,
                    ctypes.c_size_t(0),
                    ctypes.c_size_t(0),
                    ctypes.c_void_p(tensor.data_ptr() + offset),
                    ctypes.c_size_t(width * 4),
                    ctypes.c_size_t(width * 4),
                    ctypes.c_size_t(height),
                    ctypes.c_int(_cudaMemcpyDeviceToDevice)
                )
                if ret != 0:
                    raise RuntimeError(f"cudaMemcpy2DToArray layer {k} failed: {ret}")
        finally:
            _CUDART.cudaGraphicsUnmapResources(
                1, ctypes.byref(resource), ctypes.c_void_p(0))
