import unittest

import torch
import torch.nn.functional as F

from gui.cutie.model.modules import (
    SensoryDeepUpdater,
    SensoryUpdater,
    _recurrent_update,
    downsample_groups,
)
from gui.cutie.utils.tensor_utils import downsample_to_size


class TensorUtilsTest(unittest.TestCase):
    def test_downsample_to_size_matches_area_for_exact_ratio(self) -> None:
        x = torch.randn(2, 3, 16, 20)

        expected = F.interpolate(x, size=(4, 5), mode='area')
        actual = downsample_to_size(x, (4, 5))

        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


class SensoryUpdaterTest(unittest.TestCase):
    def test_sensory_deep_updater_matches_legacy_formula(self) -> None:
        torch.manual_seed(0)
        module = SensoryDeepUpdater(4, 4).eval()
        g = torch.randn(1, 2, 4, 3, 5)
        h = torch.randn(1, 2, 4, 3, 5)

        with torch.amp.autocast(device_type='cpu', enabled=False):
            values = module.transform(torch.cat([g.float(), h.float()], dim=2))
            expected = _recurrent_update(h.float(), values)

        actual = module(g, h)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_sensory_updater_matches_legacy_formula(self) -> None:
        torch.manual_seed(1)
        module = SensoryUpdater([4, 3, 5], mid_dim=4, sensory_dim=4).eval()
        g16 = torch.randn(1, 2, 4, 3, 5)
        g8 = torch.randn(1, 2, 3, 6, 10)
        g4 = torch.randn(1, 2, 5, 12, 20)
        h = torch.randn(1, 2, 4, 3, 5)

        fused = module.g16_conv(g16) + module.g8_conv(downsample_groups(g8, ratio=1 / 2)) + \
            module.g4_conv(downsample_groups(g4, ratio=1 / 4))
        with torch.amp.autocast(device_type='cpu', enabled=False):
            values = module.transform(torch.cat([fused.float(), h.float()], dim=2))
            expected = _recurrent_update(h.float(), values)

        actual = module([g16, g8, g4], h)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for autocast regression")
    def test_sensory_deep_updater_autocast_stays_close_to_legacy(self) -> None:
        torch.manual_seed(2)
        module = SensoryDeepUpdater(8, 8).cuda().eval()
        g = torch.randn(1, 2, 8, 4, 6, device='cuda')
        h = torch.randn(1, 2, 8, 4, 6, device='cuda')

        with torch.autocast('cuda', dtype=torch.float16):
            actual = module(g, h)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            values = module.transform(torch.cat([g.float(), h.float()], dim=2))
            expected = _recurrent_update(h.float(), values)

        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for autocast regression")
    def test_sensory_updater_autocast_stays_close_to_legacy(self) -> None:
        torch.manual_seed(3)
        module = SensoryUpdater([8, 6, 10], mid_dim=8, sensory_dim=8).cuda().eval()
        g16 = torch.randn(1, 2, 8, 4, 6, device='cuda')
        g8 = torch.randn(1, 2, 6, 8, 12, device='cuda')
        g4 = torch.randn(1, 2, 10, 16, 24, device='cuda')
        h = torch.randn(1, 2, 8, 4, 6, device='cuda')

        with torch.autocast('cuda', dtype=torch.float16):
            actual = module([g16, g8, g4], h)
        fused = module.g16_conv(g16) + module.g8_conv(downsample_groups(g8, ratio=1 / 2)) + \
            module.g4_conv(downsample_groups(g4, ratio=1 / 4))
        with torch.amp.autocast(device_type='cuda', enabled=False):
            values = module.transform(torch.cat([fused.float(), h.float()], dim=2))
            expected = _recurrent_update(h.float(), values)

        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


if __name__ == '__main__':
    unittest.main()
