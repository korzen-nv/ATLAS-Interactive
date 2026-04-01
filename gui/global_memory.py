"""Persistent cross-video memory store for image/mask exemplar pairs.

Items are organised on disk by workspace (source video) name.  Each item
is stored as a flat triplet of files using the original frame name::

    global_memory/
      episode_001.mp4/
        frame42.jpg   frame42.png   frame42.json
        frame99.jpg   frame99.png   frame99.json
      episode_002.mp4/
        frame10.jpg   frame10.png   frame10.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class GlobalMemoryItem:
    """A single image/mask pair stored in global memory."""
    name: str
    source_video: str
    frame_idx: int
    path: Path          # folder containing the triplet

    @property
    def image_path(self) -> Path:
        return self.path / f'{self.name}.jpg'

    @property
    def mask_path(self) -> Path:
        return self.path / f'{self.name}.png'

    @property
    def meta_path(self) -> Path:
        return self.path / f'{self.name}.json'

    def load_image(self) -> np.ndarray:
        img = cv2.imread(str(self.image_path))
        return img[:, :, ::-1].copy()

    def load_mask(self) -> np.ndarray:
        return np.array(Image.open(self.mask_path))


class GlobalMemoryStore:
    """Manages a persistent directory of image/mask exemplar pairs,
    organised into sub-folders by workspace (source video) name."""

    def __init__(self, store_dir: str) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # -- queries ---------------------------------------------------------------

    def folders(self) -> List[Tuple[str, int]]:
        """Return ``[(folder_name, item_count), ...]`` sorted by name."""
        result = []
        if not self.store_dir.exists():
            return result
        for d in sorted(self.store_dir.iterdir()):
            if not d.is_dir():
                continue
            count = sum(1 for x in d.glob('*.json'))
            if count > 0:
                result.append((d.name, count))
        return result

    def folder_items(self, folder_name: str) -> List[GlobalMemoryItem]:
        """Return all items inside *folder_name*."""
        folder = self.store_dir / folder_name
        items: List[GlobalMemoryItem] = []
        if not folder.is_dir():
            return items
        for meta_path in sorted(folder.glob('*.json')):
            with open(meta_path) as f:
                meta = json.load(f)
            items.append(GlobalMemoryItem(
                name=meta.get('name', meta_path.stem),
                source_video=meta.get('source_video', folder_name),
                frame_idx=meta.get('frame_idx', -1),
                path=folder,
            ))
        return items

    def all_items(self) -> List[GlobalMemoryItem]:
        """Return every item across all folders."""
        items: List[GlobalMemoryItem] = []
        for folder_name, _ in self.folders():
            items.extend(self.folder_items(folder_name))
        return items

    def total_count(self) -> int:
        return sum(c for _, c in self.folders())

    # -- mutations -------------------------------------------------------------

    def add(
        self,
        name: str,
        image: np.ndarray,
        mask: np.ndarray,
        source_video: str = '',
        frame_idx: int = -1,
        palette=None,
    ) -> GlobalMemoryItem:
        """Save an image/mask pair under ``source_video/`` sub-folder."""
        folder = self.store_dir / source_video
        folder.mkdir(parents=True, exist_ok=True)

        safe_name = name.replace('/', '_').replace('\\', '_').replace(' ', '_')

        Image.fromarray(image).save(folder / f'{safe_name}.jpg', quality=95)

        mask_img = Image.fromarray(mask.astype(np.uint8))
        if palette is not None:
            mask_img.putpalette(palette)
        mask_img.save(folder / f'{safe_name}.png')

        meta = {'name': safe_name, 'source_video': source_video, 'frame_idx': frame_idx}
        with open(folder / f'{safe_name}.json', 'w') as f:
            json.dump(meta, f, indent=2)

        return GlobalMemoryItem(
            name=safe_name, source_video=source_video,
            frame_idx=frame_idx, path=folder,
        )

    def remove_item(self, folder_name: str, item_name: str) -> None:
        """Remove a single item (its .jpg, .png, .json) from a folder."""
        folder = self.store_dir / folder_name
        for ext in ('.jpg', '.png', '.json'):
            p = folder / f'{item_name}{ext}'
            if p.exists():
                p.unlink()

    def remove_folder(self, folder_name: str) -> None:
        folder = self.store_dir / folder_name
        if folder.exists():
            shutil.rmtree(folder)

    def clear(self) -> None:
        for folder_name, _ in self.folders():
            self.remove_folder(folder_name)
