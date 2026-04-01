import unittest

import torch
from omegaconf import OmegaConf

from gui.cutie.model.transformer.object_transformer import QueryTransformer
from gui.cutie.model.transformer.transformer_layers import CrossAttention, SelfAttention


class TransformerLayersTest(unittest.TestCase):
    def test_self_attention_matches_reference_mha(self) -> None:
        torch.manual_seed(0)
        module = SelfAttention(8, 2, dropout=0.0)
        module.eval()

        x = torch.randn(2, 5, 8)
        pe = torch.randn(2, 5, 8)

        x_norm = module.norm(x)
        x_with_pe = x_norm + pe
        q = x_with_pe if module.add_pe_to_qkv[0] else x_norm
        k = x_with_pe if module.add_pe_to_qkv[1] else x_norm
        v = x_with_pe if module.add_pe_to_qkv[2] else x_norm
        reference, _ = module.self_attn(q, k, v, need_weights=False)

        result = module(x, pe)
        torch.testing.assert_close(result, x_norm + reference, atol=1e-6, rtol=1e-5)

    def test_cross_attention_sdpa_matches_legacy_mha_with_mask(self) -> None:
        torch.manual_seed(1)
        module = CrossAttention(8, 2, dropout=0.0)
        module.eval()

        x = torch.randn(2, 3, 8)
        mem = torch.randn(2, 6, 8)
        x_pe = torch.randn(2, 3, 8)
        mem_pe = torch.randn(2, 6, 8)
        attn_mask = torch.zeros(2, 1, 3, 6)
        attn_mask[:, :, 0, -2:] = float('-inf')

        fast, _ = module(x, mem, x_pe, mem_pe, attn_mask=attn_mask, need_weights=False)
        slow, weights = module(x, mem, x_pe, mem_pe, attn_mask=attn_mask, need_weights=True)

        torch.testing.assert_close(fast, slow, atol=1e-6, rtol=1e-5)
        self.assertEqual(weights.shape, (2, 2, 3, 6))

    def test_object_transformer_aux_mask_flattens_object_dimension(self) -> None:
        cfg = OmegaConf.create({
            'value_dim': 8,
            'embed_dim': 8,
            'pixel_pe_scale': 32,
            'pixel_pe_temperature': 128,
            'object_transformer': {
                'embed_dim': 8,
                'ff_dim': 16,
                'num_heads': 2,
                'num_blocks': 1,
                'num_queries': 4,
                'read_from_pixel': {
                    'add_pe_to_qkv': [True, True, False],
                },
                'read_from_query': {
                    'add_pe_to_qkv': [True, True, False],
                    'output_norm': False,
                },
                'query_self_attention': {
                    'add_pe_to_qkv': [True, True, False],
                },
            },
        })
        module = QueryTransformer(cfg)
        logits = torch.randn(2, 3, 4, 5)

        attn_mask = module._get_aux_mask(logits, selector=None)

        self.assertEqual(attn_mask.shape, (6, 1, 4, 20))
        self.assertTrue(torch.isinf(attn_mask).any() or torch.count_nonzero(attn_mask) == 0)


if __name__ == '__main__':
    unittest.main()
