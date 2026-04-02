import unittest
from types import SimpleNamespace

import torch

from gui.backends.cutie_backend import CutieBackend


class CutieBackendTest(unittest.TestCase):
    def test_to_canonical_prob_restores_object_id_channels(self) -> None:
        backend = CutieBackend.__new__(CutieBackend)
        backend._cfg = {'num_objects': 19}
        backend._core = SimpleNamespace(
            object_manager=SimpleNamespace(
                tmp_id_to_obj={
                    1: SimpleNamespace(id=1),
                    2: SimpleNamespace(id=3),
                    3: SimpleNamespace(id=7),
                }))

        output_prob = torch.zeros(4, 2, 2)
        output_prob[0] = 0.25
        output_prob[1] = 0.10
        output_prob[2] = 0.20
        output_prob[3] = 0.45

        canonical = backend._to_canonical_prob(output_prob)

        self.assertEqual(tuple(canonical.shape), (20, 2, 2))
        torch.testing.assert_close(canonical[0], output_prob[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(canonical[1], output_prob[1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(canonical[3], output_prob[2], atol=0.0, rtol=0.0)
        torch.testing.assert_close(canonical[7], output_prob[3], atol=0.0, rtol=0.0)
        self.assertEqual(float(canonical[2].sum()), 0.0)
        self.assertEqual(float(canonical[4].sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
