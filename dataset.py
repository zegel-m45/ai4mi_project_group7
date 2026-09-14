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
import json
import re

import numpy as np

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 require_annotations=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        self.annotations = {}
        manifest_path = Path(root_dir) / 'annotations.json'
        self.has_annotations = manifest_path.exists()
        if require_annotations and not self.has_annotations:
            raise ValueError(f"Marginal loss requires {manifest_path}; prepare the dataset with --use-gt2-patient7.")
        if self.has_annotations:
            images = sorted((Path(root_dir) / subset / 'img').glob('*.png'))
            if not images:
                raise ValueError(f"No PNG images found in {Path(root_dir) / subset / 'img'}")
            if not self.test_mode:
                labels = sorted((Path(root_dir) / subset / 'gt').glob('*.png'))
                if [p.stem for p in images] != [p.stem for p in labels]:
                    raise ValueError(f"Image and GT filenames do not match in {Path(root_dir) / subset}")
            manifest = json.loads(manifest_path.read_text())
            if (not isinstance(manifest, dict) or manifest.get('version') != 1 or
                    manifest.get('label_schema') != 'segthor_merged_1_4'):
                raise ValueError(f"Unsupported annotation schema in {manifest_path}")
            patients = manifest.get('patients')
            if not isinstance(patients, dict):
                raise ValueError(f"Missing patient annotations in {manifest_path}")
            for image_path, _ in self.files:
                if not re.fullmatch(r'Patient_\d{2}_\d{4}', image_path.stem):
                    raise ValueError(f"Unexpected SegTHOR slice filename: {image_path.name}")
                patient = image_path.stem.rsplit('_', 1)[0]
                annotation = patients.get(patient)
                if (not isinstance(annotation, dict) or annotation.get('split') != subset or
                        annotation.get('annotation') not in ('fine', 'merged') or
                        not isinstance(annotation.get('source'), str)):
                    raise ValueError(f"Missing or conflicting annotation for {subset}/{patient}")
                self.annotations[patient] = annotation
            expected = {p for p, a in patients.items()
                        if isinstance(a, dict) and a.get('split') == subset}
            if expected != set(self.annotations):
                raise ValueError(f"Patient list differs from annotation metadata for {subset}")
        if debug:
            if self.has_annotations:
                selected = []
                kinds = sorted({a['annotation'] for a in self.annotations.values()})
                for kind in kinds:
                    patient = next(p for p in sorted(self.annotations)
                                   if self.annotations[p]['annotation'] == kind)
                    candidates = [pair for pair in self.files
                                  if pair[0].stem.rsplit('_', 1)[0] == patient]
                    count = 10 // len(kinds)
                    start = max(0, (len(candidates) - count) // 2)
                    selected.extend(candidates[start:start + count])
                self.files = sorted(selected)
            else:
                self.files = self.files[:10]

        self.patient_ids = sorted({p.stem.rsplit('_', 1)[0] for p, _ in self.files})
        self.fine_patients = [p for p in self.patient_ids
                              if self.annotations.get(p, {}).get('annotation') == 'fine']

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            if self.has_annotations:
                with Image.open(gt_path) as mask:
                    values = np.asarray(mask)
                    if values.ndim != 2 or not np.isin(values, [0, 63, 126, 189, 252]).all():
                        raise ValueError(f"SegTHOR GT must encode labels 0..4 as 0,63,126,189,252: {gt_path}")
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        if self.has_annotations:
            patient = img_path.stem.rsplit('_', 1)[0]
            is_fine = self.annotations[patient]['annotation'] == 'fine'
            if not self.test_mode and (K != 5 or (not is_fine and bool(gt[4].any()))):
                raise ValueError(f"GT values conflict with annotation metadata: {gt_path}")
            data_dict['patient_ids'] = patient
            data_dict['is_fine'] = is_fine

        return data_dict
