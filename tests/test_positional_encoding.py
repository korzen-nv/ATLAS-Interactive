import unittest

import torch

from gui.cutie.model.transformer.positional_encoding import PositionalEncoding


class PositionalEncodingTest(unittest.TestCase):
    def test_cache_reuses_spatial_encoding_across_batch_sizes(self) -> None:
        pe = PositionalEncoding(8, channel_last=False, transpose_output=True)

        out_a = pe(torch.ones(2, 8, 4, 5))
        out_b = pe(torch.ones(5, 8, 4, 5))

        self.assertEqual(pe.cached_penc.shape[0], 1)
        torch.testing.assert_close(out_a[0], out_b[0], atol=0.0, rtol=0.0)

    def test_five_d_output_broadcasts_across_object_slots(self) -> None:
        pe = PositionalEncoding(8, channel_last=False, transpose_output=True)

        out = pe(torch.ones(2, 3, 8, 4, 5))

        self.assertEqual(tuple(out.shape), (2, 3, 4, 5, 8))
        torch.testing.assert_close(out[:, 0], out[:, 1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(out[:, 1], out[:, 2], atol=0.0, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
