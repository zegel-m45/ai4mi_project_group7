#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2024 Hoel Kervadec

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

import re
import argparse
from itertools import repeat
from pathlib import Path
from typing import Match, Pattern

import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
from nibabel.spaces import vox2out_vox
from skimage.io import imread
from skimage.transform import resize

from utils import map_, tqdm_

import cupy as cp
from cucim.skimage.measure import label as cucim_label

def get_z(image: Path) -> int:
    return int(image.stem.split('_')[-1])

def keep_largest_component_3d(mask_3d: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated 3D connected component using cuCIM.
    """
    # Move array to GPU (CuPy)
    mask_gpu = cp.asarray(mask_3d)
    cleaned_mask_gpu = cp.zeros_like(mask_gpu)

    # Directly copy class 1 over so it stays completely unfiltered
    cleaned_mask_gpu[mask_gpu == 1] = 1

    classes = cp.unique(mask_gpu)
    # Skip background class and esophagus (already handled)
    classes = classes[~cp.isin(classes, cp.asarray((0,1)))]
    
    for cls in classes:
        binary_mask = (mask_gpu == cls)
        
        # cucim GPU label (faster)
        # Connectivity of 1, 2 or 3 (6, 18, 26)
        labeled_mask, num_features = cucim_label(binary_mask, connectivity=2, return_num=True)
        
        if num_features == 0:
            continue
        # Faster than returning to CPU 
        sizes = cp.array([(labeled_mask == i).sum() for i in range(1, int(num_features) + 1)])
        largest_label = int(cp.argmax(sizes)) + 1
        # # Clear all voxels of this class first, then write back only the largest component
        cleaned_mask_gpu[labeled_mask == largest_label] = cls
    # Move final result back to CPU NumPy array
    return cp.asnumpy(cleaned_mask_gpu)

def merge_patient(id_: str, dest_folder: str, images: list[Path],
                  idxes: list[int], K: int, source_pattern: str, bspline: bool = False,
                  new_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    # print(source_pattern.format(id_=id_))
    orig_nib = nib.load(source_pattern.format(id_=id_))
    orig_shape = np.asarray(orig_nib.dataobj).shape
    # print(orig_nib.affine)

    # Recreate the grid used by resample_to_output in slice_segthor.py
    shape, affine = vox2out_vox((orig_shape, orig_nib.affine), new_spacing) if bspline else (orig_shape, orig_nib.affine)
    X, Y, Z = shape

    slice_indices = [get_z(images[i]) for i in idxes]
    expected_indices = list(range(Z))
    if sorted(slice_indices) != expected_indices:
        raise ValueError(f"{id_}: missing, duplicate, or unexpected slices")

    res_arr: np.ndarray = np.zeros((X, Y, Z), dtype=np.int16)

    for idx in idxes:
        img: Path = images[idx]

        z = get_z(img)
        img_arr = imread(img)
        assert img_arr.dtype == np.uint8
        assert set(np.unique(img_arr)) <= set(range(K))
        img_arr = img_arr // 63

        resized: np.ndarray = resize(img_arr, (Y, X),
                                     mode="constant",
                                     preserve_range=True,
                                     anti_aliasing=False,
                                     order=0)

        res_arr[:, :, z] = resized[...].T # transpose to match 3D slicer and online images orientation

    assert set(np.unique(res_arr)) <= set(range(K))

    # Keep only the largest component
    res_arr = keep_largest_component_3d(res_arr)

    if bspline:
        res_arr = np.flip(res_arr, axis=(0, 1))

        res_arr = np.asarray(nibproc.resample_from_to(
            nib.Nifti1Image(res_arr, affine), orig_nib, order=0, mode='nearest').dataobj).astype(np.int16)
    
    assert orig_shape == res_arr.shape, (orig_shape, res_arr.shape)

    new_nib = nib.nifti1.Nifti1Image(res_arr, affine=orig_nib.affine, header=orig_nib.header)
    nib.save(new_nib, (Path(dest_folder) / id_).with_suffix(".nii.gz"))


def main(args) -> None:
    images: list[Path] = list(Path(args.data_folder).glob("*.png"))
    grouping_regex: Pattern = re.compile(args.grp_regex)

    stems: list[str] = map_(lambda p: p.stem, images)

    matches: list[Match] = map_(grouping_regex.match, stems)  # type: ignore
    patients: list[str] = [match.group(1) for match in matches]
    unique_patients: list[str] = list(set(patients))
    print(unique_patients)
    assert len(unique_patients) < len(images)
    print(f"Found {len(unique_patients)} unique patients out of {len(images)} images ; regex: {args.grp_regex}")

    idx_map: dict[str, list[int]] = dict(zip(unique_patients, repeat(None)))  # type: ignore
    for i, patient in enumerate(patients):
        if not idx_map[patient]:
            idx_map[patient] = []

        idx_map[patient] += [i]

    # print(idx_map)
    assert sum(len(idx_map[k]) for k in unique_patients) == len(images)

    args.dest_folder.mkdir(parents=True, exist_ok=True)

    for p in tqdm_(unique_patients):
        merge_patient(p, args.dest_folder, images, idx_map[p], args.num_classes, args.source_scan_pattern, args.bspline, tuple(args.new_spacing))
    # mmap_(lambda p: merge_patient(p, args.dest_folder, images, idx_map[p], K=args.num_classes), patients)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merging slices parameters')
    parser.add_argument('--data_folder', type=Path, required=True,
                        help="The folder containing the images to predict")
    parser.add_argument('--source_scan_pattern', type=str, required=True,
                        help="The pattern to get the original scan. This is used to get the correct metadata")
    parser.add_argument('--dest_folder', type=Path, required=True)
    parser.add_argument('--grp_regex', type=str, required=True)

    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--new_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Voxel spacing used when slicing with --bspline.")
    parser.add_argument('--bspline', action='store_true',
                        help="When stitching data was sliced with B-spline-resampled data.")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
