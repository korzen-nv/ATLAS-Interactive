import base64
import cv2
import numpy as np


def encode_frame_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode a numpy RGB image to JPEG bytes."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def encode_frame_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode a numpy RGB image to a base64 JPEG string."""
    return base64.b64encode(encode_frame_jpeg(image, quality)).decode('ascii')


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Encode a numpy mask to PNG bytes."""
    _, buf = cv2.imencode('.png', mask)
    return buf.tobytes()
