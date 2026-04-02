import unittest

from contextlib import contextmanager

import torch

from gui.cutie.inference.inference_core import InferenceCore, _infer_nonempty_prob_mask_objects
from gui.cutie.inference.object_manager import ObjectManager


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

    def test_infer_nonempty_prob_mask_objects_preserves_tracked_zero_channels(self) -> None:
        mask = torch.zeros(6, 2, 2)
        mask[0, 0, 0] = 1.0
        mask[1, 0, 1] = 1.0

        objects, filtered = _infer_nonempty_prob_mask_objects(mask, preserve_objects=[2, 1, 6])

        self.assertEqual(objects, [1, 2, 6])
        self.assertEqual(tuple(filtered.shape), (3, 2, 2))
        torch.testing.assert_close(filtered[0], mask[0], atol=0.0, rtol=0.0)
        torch.testing.assert_close(filtered[1], mask[1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(filtered[2], mask[5], atol=0.0, rtol=0.0)


class InferenceCoreSensoryInitTest(unittest.TestCase):
    def test_segment_reinitializes_missing_sensory_before_memory_read(self) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.engaged = True
                self.initialized = []

            def initialize_sensory_if_needed(self, sample_key: torch.Tensor, ids: list[int]) -> None:
                self.initialized.append((tuple(ids), tuple(sample_key.shape)))

            def read(self, pix_feat, key, selection, last_mask, network, readout_fn=None, profiler=None):
                assert self.initialized, "sensory must be initialized before memory read"
                return {1: torch.zeros(1, 4, 2, 2)}

            def get_sensory(self, ids: list[int]) -> torch.Tensor:
                assert self.initialized, "sensory must be initialized before get_sensory"
                return torch.zeros(1, len(ids), 3, 2, 2)

            def update_sensory(self, sensory: torch.Tensor, ids: list[int]) -> None:
                self.updated = (tuple(ids), tuple(sensory.shape))

        class FakeObjectManager:
            all_obj_ids = [1]

            def realize_dict(self, obj_dict, dim=1) -> torch.Tensor:
                return torch.stack([obj_dict[1]], dim=dim)

        class FakeNetwork:
            def segment(self, ms_features, memory_readout, current_sensory, chunk_size, update_sensory):
                sensory = torch.zeros(1, 1, 3, 2, 2)
                pred_prob = torch.zeros(1, 2, 8, 8)
                return sensory, None, pred_prob

        class FakeProfiler:
            def mark(self, label: str) -> None:
                return None

            @contextmanager
            def section(self, label: str):
                yield

        core = InferenceCore.__new__(InferenceCore)
        core.flip_aug = False
        core.chunk_size = 1
        core.last_mask = torch.zeros(1, 1, 2, 2)
        core.memory = FakeMemory()
        core.object_manager = FakeObjectManager()
        core.network = FakeNetwork()
        core.trt_mask_decoder = None
        core.trt_readout = None
        core.profiler = FakeProfiler()

        key = torch.zeros(1, 8, 2, 2)
        selection = torch.zeros(1, 8, 2, 2)
        pix_feat = torch.zeros(1, 4, 2, 2)
        ms_features = [torch.zeros(1, 4, 2, 2) for _ in range(3)]

        pred_prob = core._segment(key, selection, pix_feat, ms_features, update_sensory=True)

        self.assertEqual(core.memory.initialized, [((1,), (1, 8, 2, 2))])
        self.assertEqual(tuple(pred_prob.shape), (2, 8, 8))

    def test_step_reorders_probability_mask_from_object_order_to_tmp_order(self) -> None:
        class FakeImageFeatureStore:
            def get_features(self, curr_ti: int, image: torch.Tensor):
                feat = torch.zeros(1, 4, 1, 1)
                return [feat, feat, feat], feat

            def get_key(self, curr_ti: int, image: torch.Tensor):
                key = torch.zeros(1, 8, 1, 1)
                shrinkage = torch.zeros(1, 1, 1, 1)
                selection = torch.zeros(1, 8, 1, 1)
                return key, shrinkage, selection

            def delete(self, curr_ti: int) -> None:
                return None

        class FakeProfiler:
            def mark(self, label: str) -> None:
                return None

            @contextmanager
            def section(self, label: str):
                yield

            def finish_frame(self) -> None:
                return None

        core = InferenceCore.__new__(InferenceCore)
        core.flip_aug = False
        core.max_internal_size = 0
        core.torch_rt = None
        core.curr_ti = -1
        core.last_mem_ti = 0
        core.mem_every = 5
        core.stagger_ti = set()
        core.chunk_size = 1
        core.profiler = FakeProfiler()
        core.image_feature_store = FakeImageFeatureStore()
        core.object_manager = ObjectManager()
        core.object_manager.add_new_objects([2, 1])
        core.last_mask = torch.zeros(1, 2, 16, 16)

        old_pred = torch.zeros(3, 16, 16)
        core._segment = lambda key, selection, pix_feat, ms_feat, update_sensory=True: old_pred

        image = torch.zeros(3, 16, 16)
        mask = torch.zeros(3, 16, 16)
        mask[0, 0, 0] = 10.0  # object 1
        mask[1, 0, 1] = 10.0  # object 2
        mask[2, 0, 2] = 10.0  # object 3 (new)

        output_prob = core.step(
            image,
            mask,
            objects=[1, 2, 3],
            idx_mask=False,
            end=True,
        )
        output_mask = core.output_prob_to_mask(output_prob)

        self.assertEqual(int(output_mask[0, 0]), 1)
        self.assertEqual(int(output_mask[0, 1]), 2)
        self.assertEqual(int(output_mask[0, 2]), 3)

    def test_step_keeps_tracked_zero_channel_for_removed_object(self) -> None:
        class FakeImageFeatureStore:
            def get_features(self, curr_ti: int, image: torch.Tensor):
                feat = torch.zeros(1, 4, 1, 1)
                return [feat, feat, feat], feat

            def get_key(self, curr_ti: int, image: torch.Tensor):
                key = torch.zeros(1, 8, 1, 1)
                shrinkage = torch.zeros(1, 1, 1, 1)
                selection = torch.zeros(1, 8, 1, 1)
                return key, shrinkage, selection

            def delete(self, curr_ti: int) -> None:
                return None

        class FakeProfiler:
            def mark(self, label: str) -> None:
                return None

            @contextmanager
            def section(self, label: str):
                yield

            def finish_frame(self) -> None:
                return None

        core = InferenceCore.__new__(InferenceCore)
        core.flip_aug = False
        core.max_internal_size = 0
        core.torch_rt = None
        core.curr_ti = -1
        core.last_mem_ti = 0
        core.mem_every = 5
        core.stagger_ti = set()
        core.chunk_size = 1
        core.profiler = FakeProfiler()
        core.image_feature_store = FakeImageFeatureStore()
        core.object_manager = ObjectManager()
        core.object_manager.add_new_objects([2, 1, 6])
        core.last_mask = torch.zeros(1, 3, 16, 16)
        core._segment = lambda key, selection, pix_feat, ms_feat, update_sensory=True: torch.zeros(
            4, 16, 16)

        image = torch.zeros(3, 16, 16)
        mask = torch.zeros(6, 16, 16)
        mask[0, 0, 0] = 10.0  # object 1
        mask[1, 0, 1] = 10.0  # object 2
        # object 6 intentionally left all-zero to simulate removal

        output_prob = core.step(
            image,
            mask,
            objects=None,
            idx_mask=False,
            end=True,
        )
        output_mask = core.output_prob_to_mask(output_prob)

        self.assertEqual(int(output_mask[0, 0]), 1)
        self.assertEqual(int(output_mask[0, 1]), 2)
        self.assertEqual(int(output_mask[0, 2]), 0)

if __name__ == '__main__':
    unittest.main()
