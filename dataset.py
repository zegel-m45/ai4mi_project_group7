#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

from torch import Tensor, randn_like
import re
import torch
from PIL import Image
import numpy as np # added for npy images produced by z-scoring
from torch.utils.data import Dataset

# ERROR FIX
def gt_stem_for(image_stem: str) -> str:
    """
    Noise-only augmentations don't modify labels, so they share the GT of the
    original slice. '_elastic_noise' is NOT noise-only (the GT was deformed),
    so it keeps its own GT file.
    """
    if image_stem.endswith("_noise") and not image_stem.endswith("_elastic_noise"):
        return image_stem[:-len("_noise")]
    return image_stem

class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False, 
                 exclude: set[str] | None = None, context_slices: int = 1):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.context_slices = context_slices
        self.context_radius = context_slices // 2
        
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError(
                        "context_slices must be a positive odd number, such as 1, 3, or 5."
            )
        
        self.files = self._make_dataset(root_dir, subset, exclude=exclude)

        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def _make_dataset(self, root, subset, exclude: set[str] | None = None) -> list[tuple[Path, Path | None]]:
        assert subset in ['train', 'val', 'test']

        root = Path(root)
        print(f"> {root=}")

        img_path = root / subset / 'img'
        full_path = root / subset / 'gt'

        images: list[Path] = sorted([*img_path.glob("*.png"), *img_path.glob("*.npy")]) # added npy images produced by z-scoring
        context_paths: list[list[Path]] = []

        ##Idea: change images here to [centre-i, ..., centre-1, image, centre+1, ..., centre+i]
        patient_slices: dict[tuple[str, str], list[Path]] = {}

        for img_path in images:
            series_id = self._series_id(img_path)
            patient_slices.setdefault(series_id, []).append(img_path)

        context_paths: dict[Path, list[Path]] = {}

        for _, image_paths in patient_slices.items():
            image_paths = sorted(image_paths, key=self._slice_number)

            for centre_index, centre_path in enumerate(image_paths):
                neighbours = []


                for offset in range (-self.context_radius, self.context_radius + 1):
                    neighbour_index = centre_index + offset

                    neighbour_index = max (0, min(neighbour_index, len(image_paths)-1))

                    neighbours.append(image_paths[neighbour_index])
                
                context_paths[centre_path] = neighbours

        
        ## ensure that context_paths is in the same order as images, so that labels are also still valid
        context_imgs: list[Path] = []
        for img_path in images:
            context_imgs.append(context_paths[img_path])

        self.context_paths = context_paths

        full_labels: list[Path | None]
        if subset != 'test':
            full_labels = [full_path / f"{gt_stem_for(image.stem)}.png" for image in images] # go through images produced by z-scoring as well
        else:
            full_labels = [None] * len(images)

               
        # Added for patient slices we need to exclude
        if exclude:
            pairs = [(img, gt) for img, gt in zip(images, full_labels) if img.stem not in exclude]
            removed = len(images) - len(pairs)
            print(f"> Excluded {removed} mislabeled slice(s) from {subset}")
            return pairs

        return list(zip(context_imgs, full_labels))

    @staticmethod
    def _load_image(img_path: Path):
        if img_path.suffix == '.npy':
            return np.load(img_path, allow_pickle=False)

        return Image.open(img_path)

    @staticmethod
    def _patient_id(img_path: Path) -> str:
        #Go from Patient_03_0080.npy to Patient_03
        return img_path.stem.rsplit('_', 1)[0]

    @staticmethod
    def _series_id(img_path: Path) -> tuple[str, str]:
        """
        Return a key identifying one coherent slice series.

        Examples:
            Patient_05_0032.npy          -> ("Patient_05", "original")
            Patient_05_0032_elastic.npy  -> ("Patient_05", "elastic")
        """
        stem = img_path.stem

        match = re.fullmatch(
            r"(?P<patient>Patient_\d+)_(?P<slice>\d+)(?:_(?P<variant>elastic))?",
            stem,
        )

        if match is None:
            raise ValueError(
                f"Unexpected image filename format: {img_path.name}. "
                "Expected e.g. Patient_05_0032.npy or "
                "Patient_05_0032_elastic.npy."
            )

        patient = match.group("patient")
        variant = match.group("variant") or "original"

        return patient, variant

    @staticmethod
    def _slice_number(img_path: Path) -> int:
        #gets slice number: Patient_03_0080 -> 80
        #return img_path.stem.rsplit('_', 1)[1]

        match = re.fullmatch(
            r"Patient_\d+_(?P<slice>\d+)(?:_elastic)?",
            img_path.stem,
        )

        if match is None:
            raise ValueError(
                f"Unexpected image filename format: {img_path.name}."
            )

        return int(match.group("slice"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        images, gt_path = self.files[index]

        slice_tensors = [
            self.img_transform(self._load_image(neighbour_path))
            for neighbour_path in images
        ]

        #concatenate [1, H, W] tensors into [context_slices, H, W].
        img: Tensor = torch.cat(slice_tensors, dim=0)

        assert img.shape[0] == self.context_slices, (
            f"Expected {self.context_slices} input channels, got {img.shape[0]} instead."
        )

        #img: Tensor = self.img_transform(np.load(img_path, allow_pickle=False) if img_path.suffix == '.npy' else Image.open(img_path)) # added npy for images produced by z-scoring

        data_dict = {"images": img,
                     "stems": images[self.context_radius].stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, gt_w, gt_h = gt.shape

            assert (gt_h, gt_w) == (H, W), (
                f"Image and GT sizes differ: image={(H, W)}, gt={(gt_h, gt_w)}."
            )

            data_dict["gts"] = gt

        return data_dict
