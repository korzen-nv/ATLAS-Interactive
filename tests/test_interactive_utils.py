import unittest

import numpy as np
import torch

from gui.interactive_utils import torch_mask_to_numpy_uint8, torch_prob_to_numpy_mask


class InteractiveUtilsTest(unittest.TestCase):
    def test_torch_mask_to_numpy_uint8_matches_existing_uint8_semantics(self) -> None:
        mask = torch.tensor([[0, 1, 255], [2, 3, 4]], dtype=torch.int64)

        mask_np = torch_mask_to_numpy_uint8(mask)

        self.assertEqual(mask_np.dtype, np.uint8)
        np.testing.assert_array_equal(mask_np, mask.numpy().astype(np.uint8))

    def test_torch_prob_to_numpy_mask_returns_uint8_argmax(self) -> None:
        prob = torch.tensor(
            [[[0.9, 0.1], [0.1, 0.2]],
             [[0.1, 0.8], [0.7, 0.1]],
             [[0.0, 0.1], [0.2, 0.7]]],
            dtype=torch.float32,
        )

        mask_np = torch_prob_to_numpy_mask(prob)

        self.assertEqual(mask_np.dtype, np.uint8)
        np.testing.assert_array_equal(mask_np, np.array([[0, 1], [1, 2]], dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
