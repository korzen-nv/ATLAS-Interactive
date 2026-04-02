# Modified from https://github.com/seoungwugoh/ivs-demo

from typing import Literal, List
import numpy as np

import torch
import torch.nn.functional as F
from gui.cutie.utils.palette import custom_palette


def image_to_torch(frame: np.ndarray, device: str = 'cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
    return frame


def torch_prob_to_numpy_mask(prob: torch.Tensor):
    mask = torch.max(prob, dim=0).indices
    return torch_mask_to_numpy_uint8(mask)


def torch_mask_to_numpy_uint8(mask: torch.Tensor):
    if mask.dtype != torch.uint8:
        mask = mask.to(dtype=torch.uint8)
    return mask.cpu().numpy()


def torch_prob_to_numpy_mask_weighted(prob: torch.Tensor,
                                      weights: torch.Tensor,
                                      mode: str = 'multiply') -> np.ndarray:
    """Convert probabilities to a uint8 mask after applying class weights."""
    w = weights.to(prob.device).view(-1, 1, 1)
    if mode == 'exponent':
        adjusted = prob.clamp(min=1e-7) ** w
    else:
        adjusted = prob * w
    mask = torch.max(adjusted, dim=0).indices
    return mask.cpu().numpy().astype(np.uint8)


def index_numpy_to_one_hot_torch(mask: np.ndarray, num_classes: int):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


"""
Some constants fro visualization
"""
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except:
    device = torch.device("cpu")

_raw_map = np.frombuffer(custom_palette, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
_raw_map = (_raw_map.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)
# pad to 256 entries so stale masks with out-of-range class IDs don't crash
color_map_np = np.zeros((256, 3), dtype=np.uint8)
color_map_np[:len(_raw_map)] = _raw_map
color_map = color_map_np.tolist()
color_map_torch = torch.from_numpy(color_map_np).to(device) / 255

grayscale_weights = np.array([[0.3, 0.59, 0.11]]).astype(np.float32)
grayscale_weights_torch = torch.from_numpy(grayscale_weights).to(device).unsqueeze(0)


def get_visualization(mode: Literal['image', 'mask', 'fade', 'davis', 'light', 'popup', 'layer',
                                    'rgba', 'soft'],
                      image: np.ndarray, mask: np.ndarray, layer: np.ndarray,
                      target_objects: List[int],
                      prob_np: np.ndarray = None,
                      selected_obj: int = 1) -> np.ndarray:
    if mode == 'image':
        return image
    elif mode == 'mask':
        return color_map_np[mask]
    elif mode == 'fade':
        return overlay_davis(image, mask, fade=True)
    elif mode == 'davis':
        return overlay_davis(image, mask)
    elif mode == 'light':
        return overlay_davis(image, mask, 0.9)
    elif mode == 'popup':
        return overlay_popup(image, mask, target_objects)
    elif mode == 'layer':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis(image, mask)
        else:
            return overlay_layer(image, mask, layer, target_objects)
    elif mode == 'rgba':
        return overlay_rgba(image, mask, target_objects)
    elif mode == 'soft':
        return overlay_soft(image, mask, prob_np, selected_obj)
    else:
        raise NotImplementedError


def get_visualization_torch(mode: Literal['image', 'mask', 'fade', 'davis', 'light', 'popup',
                                          'layer', 'rgba', 'soft'],
                            image: torch.Tensor, prob: torch.Tensor,
                            layer: torch.Tensor, target_objects: List[int],
                            selected_obj: int = 1) -> np.ndarray:
    if mode == 'image':
        return (image.permute(1, 2, 0) * 255).byte().cpu().numpy()
    elif mode == 'mask':
        mask = torch.max(prob, dim=0).indices
        return (color_map_torch[mask] * 255).byte().cpu().numpy()
    elif mode == 'fade':
        return overlay_davis_torch(image, prob, fade=True)
    elif mode == 'davis':
        return overlay_davis_torch(image, prob)
    elif mode == 'light':
        return overlay_davis_torch(image, prob, 0.9)
    elif mode == 'popup':
        return overlay_popup_torch(image, prob, target_objects)
    elif mode == 'layer':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis_torch(image, prob)
        else:
            return overlay_layer_torch(image, prob, layer, target_objects)
    elif mode == 'rgba':
        return overlay_rgba_torch(image, prob, target_objects)
    elif mode == 'soft':
        return overlay_soft_torch(image, prob, selected_obj)
    else:
        raise NotImplementedError


def overlay_mask_diff(image: np.ndarray, prev_mask: np.ndarray, curr_mask: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    """Overlay showing how masks changed between two frames.

    Args:
        image: (H, W, 3) uint8 RGB image (current frame visualization).
        prev_mask: (H, W) uint8 mask from the neighboring frame.
        curr_mask: (H, W) uint8 mask from the current frame.
        alpha: Blending strength of the diff overlay.

    Returns:
        (H, W, 3) uint8 composited image with colored diff overlay.
        Green = added, Red = removed, Yellow = class changed.
    """
    prev_fg = prev_mask > 0
    curr_fg = curr_mask > 0

    added = ~prev_fg & curr_fg          # new mask pixels
    removed = prev_fg & ~curr_fg        # lost mask pixels
    changed = prev_fg & curr_fg & (prev_mask != curr_mask)  # class swap

    overlay = np.zeros_like(image, dtype=np.float32)
    blend = np.zeros(image.shape[:2], dtype=np.float32)

    # green for added
    overlay[added] = [0, 255, 0]
    blend[added] = alpha
    # red for removed
    overlay[removed] = [255, 0, 0]
    blend[removed] = alpha
    # yellow for class changed
    overlay[changed] = [255, 255, 0]
    blend[changed] = alpha

    blend = blend[:, :, np.newaxis]
    result = image * (1 - blend) + overlay * blend
    return result.clip(0, 255).astype(np.uint8)


def overlay_change_heatmap(image: np.ndarray, heatmap: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    """Blend a spatial change heatmap onto an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        heatmap: (H, W) float [0, 1] normalized distance map.
        alpha: Maximum blending strength for hottest regions.

    Returns:
        (H, W, 3) uint8 composited image with warm-color overlay.
    """
    # hot colormap: black → red → yellow → white
    r = np.clip(heatmap * 3, 0, 1)
    g = np.clip(heatmap * 3 - 1, 0, 1)
    b = np.clip(heatmap * 3 - 2, 0, 1)
    color = np.stack([r, g, b], axis=-1) * 255  # (H, W, 3)

    blend = (heatmap * alpha)[:, :, np.newaxis]  # per-pixel blend weight
    result = image * (1 - blend) + color * blend
    return result.clip(0, 255).astype(np.uint8)


def overlay_davis(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, fade: bool = False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)


def overlay_popup(image: np.ndarray, mask: np.ndarray, target_objects: List[int]):
    # Keep foreground colored. Convert background to grayscale.
    im_overlay = image.copy()

    binary_mask = ~(np.isin(mask, target_objects))
    colored_region = (im_overlay[binary_mask] * grayscale_weights).sum(-1, keepdims=-1)
    im_overlay[binary_mask] = colored_region
    return im_overlay.astype(image.dtype)


def overlay_layer(image: np.ndarray, mask: np.ndarray, layer: np.ndarray,
                  target_objects: List[int]):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    obj_mask = (np.isin(mask, target_objects)).astype(np.float32)[:, :, np.newaxis]
    layer_alpha = layer[:, :, 3].astype(np.float32)[:, :, np.newaxis] / 255
    layer_rgb = layer[:, :, :3]
    background_alpha = (1 - obj_mask) * (1 - layer_alpha)
    im_overlay = (image * background_alpha + layer_rgb * (1 - obj_mask) * layer_alpha +
                  image * obj_mask).clip(0, 255)
    return im_overlay.astype(image.dtype)


def overlay_rgba(image: np.ndarray, mask: np.ndarray, target_objects: List[int]):
    # Put the mask is in the alpha channel
    obj_mask = (np.isin(mask, target_objects)).astype(np.float32)[:, :, np.newaxis] * 255
    im_overlay = np.concatenate([image, obj_mask], axis=-1)
    return im_overlay.astype(image.dtype)


def overlay_davis_torch(image: torch.Tensor,
                        prob: torch.Tensor,
                        alpha: float = 0.5,
                        fade: bool = False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    # NOTE: Make sure you no longer use image after calling this function
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.max(prob, dim=0).indices

    colored_mask = color_map_torch[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_popup_torch(image: torch.Tensor, prob: torch.Tensor, target_objects: List[int]):
    # Keep foreground colored. Convert background to grayscale.
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # I should not need to convert this to numpy.
        # Using list works most of the time but consistently fails
        # if I include first object -> exclude it -> include it again.
        # I check everywhere and it makes absolutely no sense.
        # I am blaming this on PyTorch and calling it a day
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)
    gray_image = (image * grayscale_weights_torch).sum(-1, keepdim=True)
    im_overlay = obj_mask * image + (1 - obj_mask) * gray_image

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_layer_torch(image: torch.Tensor, prob: torch.Tensor, layer: torch.Tensor,
                        target_objects: List[int]):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # TODO: figure out why we need to convert this to numpy array
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)
    layer_alpha = layer[:, :, 3].unsqueeze(2)
    layer_rgb = layer[:, :, :3]
    # background_alpha = torch.maximum(obj_mask, layer_alpha)
    background_alpha = (1 - obj_mask) * (1 - layer_alpha)
    im_overlay = (image * background_alpha + layer_rgb * (1 - obj_mask) * layer_alpha +
                  image * obj_mask).clip(0, 1)

    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_rgba_torch(image: torch.Tensor, prob: torch.Tensor, target_objects: List[int]):
    image = image.permute(1, 2, 0)

    if len(target_objects) == 0:
        obj_mask = torch.zeros_like(prob[0]).unsqueeze(2)
    else:
        # TODO: figure out why we need to convert this to numpy array
        obj_mask = prob[np.array(target_objects, dtype=np.int32)].sum(0).unsqueeze(2)

    im_overlay = torch.cat([image, obj_mask], dim=-1).clip(0, 1)
    im_overlay = (im_overlay * 255).byte().cpu().numpy()
    return im_overlay


def overlay_soft(image: np.ndarray, mask: np.ndarray, prob_np: np.ndarray,
                 selected_obj: int) -> np.ndarray:
    """Soft-mask overlay: tint the selected class using its probability as alpha.

    Args:
        image: (H, W, 3) uint8.
        mask: (H, W) uint8 — unused, kept for API symmetry.
        prob_np: (C, H, W) float32 probabilities, or None.
        selected_obj: 1-based class index.
    """
    if prob_np is None or selected_obj < 1 or selected_obj >= prob_np.shape[0]:
        return image.copy()
    alpha = prob_np[selected_obj]  # (H, W) float in [0,1]
    r, g, b = color_map_np[selected_obj]
    tint = np.array([r, g, b], dtype=np.float32) / 255.0
    img_f = image.astype(np.float32) / 255.0
    blended = img_f * (1 - alpha[:, :, None] * 0.5) + tint[None, None, :] * alpha[:, :, None] * 0.5
    return (blended.clip(0, 1) * 255).astype(np.uint8)


def overlay_soft_torch(image: torch.Tensor, prob: torch.Tensor,
                       selected_obj: int) -> np.ndarray:
    """Soft-mask overlay (GPU path): tint the selected class using its probability."""
    image = image.permute(1, 2, 0)  # (H, W, 3)
    if selected_obj < 1 or selected_obj >= prob.shape[0]:
        return (image * 255).byte().cpu().numpy()
    alpha = prob[selected_obj].unsqueeze(2)  # (H, W, 1) soft probability
    color = color_map_torch[selected_obj].view(1, 1, 3)  # class color
    blended = image * (1 - alpha * 0.5) + color * alpha * 0.5
    return (blended.clip(0, 1) * 255).byte().cpu().numpy()
