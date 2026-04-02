import unittest

import torch

from gui.cutie.inference.inference_core import _infer_nonempty_prob_mask_objects


class InferenceCoreMaskInferenceTest(unittest.TestCase):
    def test_infer_nonempty_prob_mask_objects_filters_empty_channels(self) -> None:
        mask = torch.zeros(5, 4, 6)
        mask[0, 1, 2] = 1.0
        mask[2, 0, 0] = 0.3
        mask[4, 3, 5] = 0.7

        objects, filtered = _infer_nonempty_prob_mask_objects(mask)

        self.assertEqual(objects, [1, 3, 5])
        self.assertEqual(tuple(filtered.shape), (3, 4, 6))
        torch.testing.assert_close(filtered[0], mask[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(filtered[1], mask[2], atol=0.0, rtol=0.0)
        torch.testing.assert_close(filtered[2], mask[4], atol=0.0, rtol=0.0)

    def test_infer_nonempty_prob_mask_objects_keeps_all_channels_when_none_are_empty(self) -> None:
        mask = torch.rand(3, 2, 2)

        objects, filtered = _infer_nonempty_prob_mask_objects(mask)

        self.assertEqual(objects, [1, 2, 3])
        self.assertEqual(tuple(filtered.shape), tuple(mask.shape))
        torch.testing.assert_close(filtered, mask, atol=0.0, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
