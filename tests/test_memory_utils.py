import unittest

import torch

from gui.cutie.model.utils.memory_utils import (
    _chunked_topk_softmax_sparse,
    do_softmax_sparse,
    get_similarity,
    sparse_topk_affinity,
)


def _dense_from_sparse(weights: torch.Tensor,
                       indices: torch.Tensor,
                       num_tokens: int) -> torch.Tensor:
    dense = torch.zeros(weights.shape[0],
                        num_tokens,
                        weights.shape[-1],
                        dtype=weights.dtype,
                        device=weights.device)
    dense.scatter_(1, indices, weights)
    return dense


class SparseTopkAffinityTest(unittest.TestCase):
    def _assert_sparse_close(self, reference: tuple, candidate: tuple, num_tokens: int) -> None:
        ref_weights, ref_indices = reference[:2]
        cand_weights, cand_indices = candidate[:2]
        ref_dense = _dense_from_sparse(ref_weights, ref_indices, num_tokens)
        cand_dense = _dense_from_sparse(cand_weights, cand_indices, num_tokens)
        torch.testing.assert_close(cand_dense, ref_dense, atol=1e-5, rtol=1e-5)

        if len(reference) == 3:
            torch.testing.assert_close(candidate[2], reference[2], atol=1e-5, rtol=1e-5)

    def test_chunked_matches_dense_with_selection(self) -> None:
        torch.manual_seed(0)
        mk = torch.randn(2, 4, 11)
        ms = torch.rand(2, 1, 11) + 1.0
        qk = torch.randn(2, 4, 7)
        qe = torch.sigmoid(torch.randn(2, 4, 7))
        top_k = 4

        similarity = get_similarity(mk, ms, qk, qe)
        reference = do_softmax_sparse(similarity, top_k=top_k, return_usage=True)
        candidate = _chunked_topk_softmax_sparse(mk, ms, qk, qe, top_k, return_usage=True)

        self._assert_sparse_close(reference, candidate, num_tokens=mk.shape[-1])

    def test_public_api_matches_dense_without_selection(self) -> None:
        torch.manual_seed(1)
        mk = torch.randn(1, 8, 3, 5)
        ms = torch.rand(1, 1, 3, 5) + 1.0
        qk = torch.randn(1, 8, 2, 4)
        top_k = 5

        similarity = get_similarity(mk, ms, qk, None)
        reference = do_softmax_sparse(similarity, top_k=top_k, return_usage=True)
        candidate = sparse_topk_affinity(mk,
                                         ms,
                                         qk,
                                         None,
                                         top_k=top_k,
                                         return_usage=True,
                                         backend='triton')

        self._assert_sparse_close(reference, candidate, num_tokens=mk.flatten(start_dim=2).shape[-1])


if __name__ == '__main__':
    unittest.main()
