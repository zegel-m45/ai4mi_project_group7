#!/usr/bin/env python3.7

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

import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_
import nibabel.processing as nibproc


def compute_train_data_HU_stats(source_path: Path, ids: list[str], clip: bool = False) -> dict[str, float]:
    """
    Compute the mean, std, min, and max of the Hounsfield Unit values across the training data.
    This function iterates over the provided list of patient IDs, loads their corresponding CT scans, optionally clips
    the HU values to a specified range, and computes the statistics.
    """
    min_value = float('inf')
    max_value = float('-inf')
    total_voxels = 0.0
    sum_value = 0.0
    sum_squared_value = 0.0

    for id_ in tqdm_(ids):
        ct_path: Path = source_path / "train" / id_ / f"{id_}.nii.gz"
        nib_obj = nib.load(str(ct_path))
        ct: np.ndarray = np.asarray(nib_obj.dataobj)

        assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

        if clip:
            ct = np.clip(ct, -1000, 1000)

        min_value = min(min_value, ct.min())
        max_value = max(max_value, ct.max())
        total_voxels += ct.size
        sum_value += ct.sum()
        sum_squared_value += np.square(ct.astype(np.float64)).sum()

    mean_value = sum_value / total_voxels
    std_value = np.sqrt(max(0.0, (sum_squared_value / total_voxels) - mean_value ** 2))

    print(f"Train dataset stats\n min={min_value:.1f} max={max_value:.1f} "
          f"mean={mean_value:.2f} std={std_value:.2f} HU "
          f"({total_voxels} voxels across {len(ids)} patients)")

    return {"min": float(min_value), "max": float(max_value),
            "mean": float(mean_value), "std": std_value}


def zscore_arr_fixed(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Normalize using z-score with fixed mean and standard deviation values.
    """
    casted = img.astype(np.float32)
    shifted = casted - mean
    norm = shifted / std
    return norm.astype(np.float32)


def norm_arr_fixed(img: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """
    Normalize using min-max with fixed min and max values.
    """
    casted = img.astype(np.float32)
    shifted = casted - min_value
    norm = np.clip(shifted / (max_value - min_value), 0, 1)
    res = 255 * norm

    return res.astype(np.uint8)


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False, clip: bool = False, norm="minmax", norm_scope="train_dataset", train_dataset_stats: dict | None = None,
                  bspline: bool = False, new_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[float, float, float]:

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    # CLIPPING HU ranges before normalization
    if clip:
        ct = np.clip(ct, -1000, 1000)
        nib_obj = nib.Nifti1Image(ct, affine=nib_obj.affine, header=nib_obj.header)

    gt: np.ndarray
    if not test_mode:
        # Differentiate between used groundtruths
        if id_ == "Patient_07":
            gt_filename = "GT2.nii.gz" 
        else:
             gt_filename = "GT_split.nii.gz"
        gt_path: Path = id_path / gt_filename
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
        gt_nib = nib.Nifti1Image(gt, affine=gt_nib.affine, header=gt_nib.header)
    else:
        gt_nib = nib.Nifti1Image(np.zeros_like(ct, dtype=np.uint8), affine=nib_obj.affine)
        gt = np.asarray(gt_nib.dataobj)

    # RESAMPLING USING B-SPLINE INTERPOLATION
    if bspline:
        # Already uses canonical
        ct_resampled_nib = nibproc.resample_to_output(nib_obj, voxel_sizes=new_spacing, order=3, mode='nearest')
        gt_resampled_nib = nibproc.resample_from_to(gt_nib, ct_resampled_nib, order=0, mode='nearest')

        ct = np.asarray(ct_resampled_nib.dataobj)
        gt = np.asarray(gt_resampled_nib.dataobj).astype(np.uint8)

        # Flipped images across 2 axes for the b-spline 
        ct = np.flip(ct, axis=(0, 1))
        gt = np.flip(gt, axis=(0, 1))

        # recompute
        x, y, z = ct.shape

        # Spacing of the resampled volume, before the final 2D resize
        dx, dy, dz = ct_resampled_nib.header.get_zooms()

    # NORMALIZATION
    if norm_scope == "train_dataset":
        assert train_dataset_stats is not None, "train_dataset_stats must be provided for 'train_dataset' normalization scope"
        if norm == "minmax":
            min_value = train_dataset_stats["min"]
            max_value = train_dataset_stats["max"]
            norm_ct: np.ndarray = norm_arr_fixed(ct, min_value, max_value)
        elif norm == "zscore":
            mean = train_dataset_stats["mean"]
            std = train_dataset_stats["std"]
            norm_ct: np.ndarray = zscore_arr_fixed(ct, mean, std)
        else:
            raise ValueError(f"Invalid normalization method: {norm}. Must be 'minmax' or 'zscore'.")

    elif norm_scope == "patient":
        if norm == "minmax":
            norm_ct: np.ndarray = norm_arr(ct)
        elif norm == "zscore":
            norm_ct: np.ndarray = zscore_arr_fixed(ct, ct.mean(), ct.std())
        else:
            raise ValueError(f"Invalid normalization method: {norm}. Must be 'minmax' or 'zscore'.")

    else:
        raise ValueError(f"Invalid norm_scope: {norm_scope}. Must be 'train_dataset' or 'patient'.")

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz].T, shape).astype(np.float32 if norm == "zscore" else np.uint8) # transpose to match 3D slicer and online images orientation
        gt_slice = resize_(to_slice_gt[:, :, idz].T, shape, order=0).astype(np.uint8) # transpose to match 3D slicer and online images orientation
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            if save_subfolder == "img" and norm == "zscore":
                np.save((save_path / filename).with_suffix(".npy"), data) # images produced by z-scoring are saved as .npy files as they are no longer in the range [0, 255] due to the nature of the method
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    # Saved slices are transposed then resized recording (row, column, z) in mm
    return dy * y / shape[0], dx * x / shape[1], dz


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)
    
    if args.norm_scope == "train_dataset":
        datasets_stats: dict[str, float] = compute_train_data_HU_stats(src_path, training_ids, args.clip)
    else:
        datasets_stats = None

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test',
                                 clip=args.clip,
                                 norm=args.norm,
                                 norm_scope=args.norm_scope,
                                 train_dataset_stats=datasets_stats,
                                 bspline=args.bspline,
                                 new_spacing=tuple(args.new_spacing))

        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                resolutions = list(map(pfun, iterator))
            case -1:
                resolutions = Pool().map(pfun, iterator)
            case _ as p:
                resolutions = Pool(p).map(pfun, iterator)

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")
    parser.add_argument('--clip', action='store_true',
                     help="Clip CT Hounsfield Unit values to [-1000, 1000] before normalizing.")
    parser.add_argument('--norm', type=str, default="minmax", choices=["minmax", "zscore"],
                        help="Normalization method to apply to CT images.")
    parser.add_argument('--norm_scope', type=str, default="patient", choices=["train_dataset", "patient"],
                        help="Scope of normalization: 'train_dataset' uses statistics from the training dataset, while 'patient' uses statistics from each individual patient.")
    parser.add_argument('--new_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument('--bspline', action='store_true',
                     help="Resample volumes to isotropic spacing using B-spline interpolation before slicing.")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
