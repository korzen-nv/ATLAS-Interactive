import unittest

import torch

from gui.cutie.model.utils.memory_utils import (
    _chunked_topk_softmax_sparse,
    do_softmax_sparse,
    get_similarity,
    sparse_readout,
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton path")
    def test_triton_affinity_matches_dense_on_cuda(self) -> None:
        torch.manual_seed(2)
        mk = torch.randn(1, 8, 53, device='cuda', dtype=torch.float16)
        ms = torch.rand(1, 1, 53, device='cuda', dtype=torch.float16) + 1.0
        qk = torch.randn(1, 8, 17, device='cuda', dtype=torch.float16)
        qe = torch.sigmoid(torch.randn(1, 8, 17, device='cuda', dtype=torch.float16))
        top_k = 7

        similarity = get_similarity(mk.float(), ms.float(), qk.float(), qe.float())
        reference = do_softmax_sparse(similarity, top_k=top_k, return_usage=True)
        candidate = sparse_topk_affinity(mk,
                                         ms,
                                         qk,
                                         qe,
                                         top_k=top_k,
                                         return_usage=True,
                                         backend='triton')

        ref_dense = _dense_from_sparse(reference[0], reference[1], mk.shape[-1])
        cand_dense = _dense_from_sparse(candidate[0], candidate[1], mk.shape[-1])
        torch.testing.assert_close(cand_dense, ref_dense, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(candidate[2], reference[2], atol=2e-4, rtol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton path")
    def test_triton_affinity_matches_dense_when_topk_is_power_of_two(self) -> None:
        torch.manual_seed(6)
        mk = torch.randn(1, 8, 61, device='cuda', dtype=torch.float16)
        ms = torch.rand(1, 1, 61, device='cuda', dtype=torch.float16) + 1.0
        qk = torch.randn(1, 8, 19, device='cuda', dtype=torch.float16)
        qe = torch.sigmoid(torch.randn(1, 8, 19, device='cuda', dtype=torch.float16))
        top_k = 8

        similarity = get_similarity(mk.float(), ms.float(), qk.float(), qe.float())
        reference = do_softmax_sparse(similarity, top_k=top_k, return_usage=True)
        candidate = sparse_topk_affinity(mk,
                                         ms,
                                         qk,
                                         qe,
                                         top_k=top_k,
                                         return_usage=True,
                                         backend='triton')

        ref_dense = _dense_from_sparse(reference[0], reference[1], mk.shape[-1])
        cand_dense = _dense_from_sparse(candidate[0], candidate[1], mk.shape[-1])
        torch.testing.assert_close(cand_dense, ref_dense, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(candidate[2], reference[2], atol=2e-4, rtol=2e-4)


class SparseReadoutTest(unittest.TestCase):
    def test_sparse_readout_matches_dense_cpu(self) -> None:
        torch.manual_seed(3)
        value = torch.randn(1, 3, 5, 19)
        topk_indices = torch.randint(0, 19, (1, 4, 11))
        topk_weights = torch.softmax(torch.randn(1, 4, 11), dim=1)

        expected = sparse_readout(value, topk_indices, topk_weights, backend='pytorch')
        actual = sparse_readout(value,
                                topk_indices,
                                topk_weights,
                                backend='auto',
                                v_token_major=value.permute(0, 1, 3, 2).contiguous())

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton path")
    def test_triton_sparse_readout_matches_pytorch(self) -> None:
        torch.manual_seed(4)
        value = torch.randn(1, 4, 6, 23, device='cuda', dtype=torch.float16)
        topk_indices = torch.randint(0, 23, (1, 5, 13), device='cuda')
        topk_weights = torch.softmax(torch.randn(1, 5, 13, device='cuda'), dim=1)
        value_token_major = value.permute(0, 1, 3, 2).contiguous()

        expected = sparse_readout(value, topk_indices, topk_weights, backend='pytorch')
        actual = sparse_readout(value,
                                topk_indices,
                                topk_weights,
                                backend='triton',
                                v_token_major=value_token_major)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton path")
    def test_triton_sparse_readout_accepts_token_major_only(self) -> None:
        torch.manual_seed(5)
        value = torch.randn(1, 3, 5, 17, device='cuda', dtype=torch.float16)
        topk_indices = torch.randint(0, 17, (1, 4, 9), device='cuda')
        topk_weights = torch.softmax(torch.randn(1, 4, 9, device='cuda'), dim=1)
        value_token_major = value.permute(0, 1, 3, 2).contiguous()

        expected = sparse_readout(value,
                                  topk_indices,
                                  topk_weights,
                                  backend='triton',
                                  v_token_major=value_token_major)
        actual = sparse_readout(None,
                                topk_indices,
                                topk_weights,
                                backend='triton',
                                v_token_major=value_token_major)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
