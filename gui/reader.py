import sys
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader

from gui.resource_manager import ResourceManager


class PropagationReader(Dataset):
    def __init__(self, res_man: ResourceManager, start_ti: int, direction: Literal['forward',
                                                                                   'backward']):
        self.res_man = res_man
        self.start_ti = start_ti
        self.direction = direction

        # skip the first frame
        if self.direction == 'forward':
            self.start_ti += 1
            self.length = self.res_man.T - self.start_ti
        elif self.direction == 'backward':
            self.start_ti -= 1
            self.length = self.start_ti + 1
        else:
            raise NotImplementedError

    def _frame_index(self, index: int) -> int:
        if self.direction == 'forward':
            ti = self.start_ti + index
        elif self.direction == 'backward':
            ti = self.start_ti - index
        else:
            raise NotImplementedError
        return ti

    def get_image(self, index: int):
        ti = self._frame_index(index)
        assert 0 <= ti < self.res_man.T
        return self.res_man.get_image(ti)

    def __getitem__(self, index: int):
        image = self.get_image(index)
        # Keep CPU-side propagation frames in compact uint8 form. This trims
        # DataLoader IPC volume and lets permutation/normalization happen on GPU.
        image_torch = torch.from_numpy(image)

        return image, image_torch

    def __len__(self):
        return self.length


def get_data_loader(dataset: Dataset, num_workers: int):
    if 'linux' in sys.platform:
        loader_kwargs = dict(batch_size=None,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=lambda x: x,
                             pin_memory=torch.cuda.is_available())
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 2
        loader = DataLoader(dataset, **loader_kwargs)
    else:
        print(f'Non-linux platform {sys.platform} detected, using single-threaded dataloader')
        loader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=lambda x: x)
    return loader


class PropagationPrefetcher:
    """Direct prefetcher for preloaded frame caches."""

    def __init__(self, dataset: PropagationReader, device: str, staging_slots: int = 2):
        self.dataset = dataset
        self.device = torch.device(device)
        self.length = len(dataset)
        self._cursor = 0
        self._pending = None

        self._use_cuda_prefetch = self.device.type == 'cuda'
        self._prefetch_stream = torch.cuda.Stream(device=self.device) if self._use_cuda_prefetch else None

        if self._use_cuda_prefetch:
            h = dataset.res_man.height
            w = dataset.res_man.width
            self._cpu_slots = [
                torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
                for _ in range(max(1, staging_slots))
            ]
        else:
            self._cpu_slots = None
        self._slot = 0

        if self.length > 0:
            self._pending = self._stage_index(0)

    def __iter__(self):
        return self

    def __next__(self):
        if self._pending is None:
            raise StopIteration

        current = self._pending
        self._cursor += 1
        if self._cursor < self.length:
            self._pending = self._stage_index(self._cursor)
        else:
            self._pending = None
        return current

    def _stage_index(self, index: int):
        image_np = self.dataset.get_image(index)
        cpu_tensor = torch.from_numpy(image_np)

        if self._cpu_slots is not None:
            slot = self._cpu_slots[self._slot]
            slot.copy_(cpu_tensor)
            cpu_tensor = slot
            self._slot = (self._slot + 1) % len(self._cpu_slots)

        gpu_tensor, ready_event = self._stage_tensor(cpu_tensor)
        return image_np, gpu_tensor, ready_event

    def _stage_tensor(self, cpu_tensor: torch.Tensor):
        if not self._use_cuda_prefetch:
            tensor = cpu_tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32)
            tensor.mul_(1.0 / 255.0)
            return tensor, None

        with torch.cuda.stream(self._prefetch_stream):
            tensor = cpu_tensor.to(self.device, non_blocking=True)
            tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32)
            tensor.mul_(1.0 / 255.0)
            ready_event = torch.cuda.Event()
            ready_event.record(self._prefetch_stream)

        return tensor, ready_event
