"""Dense CRF post-processing for mask boundary refinement."""
import logging
import numpy as np
import torch

log = logging.getLogger(__name__)

_HAS_CRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_CRF = True
except ImportError:
    pass


def is_available() -> bool:
    return _HAS_CRF


def apply_crf(
    image_np: np.ndarray,
    prob: torch.Tensor,
    *,
    n_iters: int = 5,
    sxy_gauss: int = 3,
    compat_gauss: int = 3,
    sxy_bilat: int = 80,
    srgb_bilat: int = 13,
    compat_bilat: int = 10,
) -> torch.Tensor:
    """Refine probabilities using Dense CRF.

    Args:
        image_np: (H, W, 3) uint8 RGB image.
        prob: (num_classes, H, W) float probability tensor (channel 0 = background).

    Returns:
        Refined (num_classes, H, W) float tensor on the same device.
    """
    if not _HAS_CRF:
        return prob

    device = prob.device
    p = np.ascontiguousarray(prob.cpu().numpy().astype(np.float32))
    n_classes, H, W = p.shape

    d = dcrf.DenseCRF2D(W, H, n_classes)
    U = np.ascontiguousarray(unary_from_softmax(p))
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gauss, compat=compat_gauss,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=sxy_bilat, srgb=srgb_bilat,
                           rgbim=np.ascontiguousarray(image_np),
                           compat=compat_bilat,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(n_iters)
    refined = np.array(Q).reshape(n_classes, H, W).astype(np.float32)
    return torch.from_numpy(refined).to(device)
