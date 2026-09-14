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
import json
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


def _check_annotation_geometry(reference, annotation, path: Path) -> None:
    if reference.shape != annotation.shape:
        raise ValueError(f"Annotation shape does not match the CT: {path}")
    if not np.allclose(reference.affine, annotation.affine, rtol=0, atol=1e-5):
        raise ValueError(f"Annotation affine does not match the CT: {path}")
    if not np.allclose(reference.header.get_zooms(), annotation.header.get_zooms(),
                       rtol=0, atol=1e-5):
        raise ValueError(f"Annotation spacing does not match the CT: {path}")


def select_annotation(id_path: Path, ct_nib, gt_nib):
    gt_path = id_path / "GT.nii.gz"
    _check_annotation_geometry(ct_nib, gt_nib, gt_path)
    gt = np.asarray(gt_nib.dataobj)
    if not set(np.unique(gt)) <= {0, 1, 2, 3}:
        raise ValueError(f"Expected merged labels 0, 1, 2, 3 in {gt_path}")
    if id_path.name != "Patient_07":
        return gt_nib

    fine_path = id_path / "GT2.nii.gz"
    if not fine_path.is_file():
        raise FileNotFoundError(f"Patient 7 fine supervision requires {fine_path}")
    fine_nib = nib.load(str(fine_path))
    _check_annotation_geometry(ct_nib, fine_nib, fine_path)
    fine = np.asarray(fine_nib.dataobj)
    fine_labels = set(np.unique(fine))
    if not fine_labels <= {0, 1, 2, 3, 4} or not {1, 4} <= fine_labels:
        raise ValueError(f"GT2 must contain esophagus (1) and aorta (4), with labels 0–4: {fine_path}")
    merged = fine.copy()
    merged[merged == 4] = 1
    if not np.array_equal(merged, gt):
        raise ValueError(f"GT2 with aorta 4 mapped to 1 must exactly reproduce {gt_path}")
    return fine_nib


def annotation_manifest(training_ids: list[str], validation_ids: list[str]) -> dict:
    if "Patient_07" not in training_ids or "Patient_07" in validation_ids:
        raise ValueError("--use-gt2-patient7 requires Patient_07 in the training split")
    patients = {}
    for split, patient_ids in (("train", training_ids), ("val", validation_ids)):
        for patient_id in patient_ids:
            fine = patient_id == "Patient_07"
            patients[patient_id] = {
                "split": split,
                "annotation": "fine" if fine else "merged",
                "source": "GT2.nii.gz" if fine else "GT.nii.gz",
            }
    return {"version": 1, "label_schema": "segthor_merged_1_4", "patients": patients}


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False, clip: bool = False, z_score: bool = False, new_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), use_gt2_patient7: bool = False) -> tuple[float, float, float]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    # Clip HU ranges before normalization 
    if clip:
        ct = np.clip(ct, -1000, 1000)
        nib_obj = nib.Nifti1Image(ct, affine=nib_obj.affine, header=nib_obj.header)

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        if use_gt2_patient7:
            gt_nib = select_annotation(id_path, nib_obj, gt_nib)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
        gt_nib = nib.Nifti1Image(gt, affine=gt_nib.affine, header=gt_nib.header)
    else:
        gt_nib = nib.Nifti1Image(np.zeros_like(ct, dtype=np.uint8), affine=nib_obj.affine)

    # Already uses canonical
    ct_resampled_nib = nibproc.resample_to_output(nib_obj, voxel_sizes=new_spacing, order=3, mode='nearest')
    gt_resampled_nib = nibproc.resample_to_output(gt_nib, voxel_sizes=new_spacing, order=0, mode='nearest')

    ct = np.asarray(ct_resampled_nib.dataobj)
    gt = np.asarray(gt_resampled_nib.dataobj).astype(np.uint8)

    # Flipped images for the b-spline
    ct = np.flip(ct, axis=(0, 1))
    gt = np.flip(gt, axis=(0, 1))

    # recompute
    x, y, z = ct.shape 

    # spacing is now new_spacing, for spacing.pkl record
    dx, dy, dz = ct_resampled_nib.header.get_zooms()  

    norm_ct: np.ndarray = norm_arr(ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(z):
        img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
        gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
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

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    return dx, dy, dz


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

    use_gt2_patient7 = getattr(args, "use_gt2_patient7", False)
    manifest = None
    if use_gt2_patient7:
        manifest = annotation_manifest(training_ids, validation_ids)
        fine_path = src_path / "train" / "Patient_07" / "GT2.nii.gz"
        if not fine_path.is_file():
            raise FileNotFoundError(f"Patient 7 fine supervision requires {fine_path}")

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
                                 new_spacing=tuple(args.new_spacing),
                                 use_gt2_patient7=use_gt2_patient7)
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

    if manifest is not None:
        with open(dest_path / "annotations.json", "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")


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
    parser.add_argument('--new_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument('--use-gt2-patient7', action='store_true',
                        help="Use validated GT2 for training Patient_07 and record fine/merged annotations")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
