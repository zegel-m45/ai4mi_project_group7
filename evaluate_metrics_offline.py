"""Evaluate saved slice PNGs or already stitched NIfTI volumes.

Examples:
python evaluate_metrics_offline.py --dimensionality 2 --pred-dir results/segthor/ce/best_epoch/val \
    --gt-dir data/SEGTHOR/val/gt --spacing-file data/SEGTHOR/spacing.pkl \
    --output-dir results/segthor/ce/hd_evaluation

python evaluate_metrics_offline.py --dimensionality 3 --pred-dir volumes/YOUR_RUN/ce \
  --gt-dir data/segthor_part1/train --gt-pattern '{patient}/GT_split.nii.gz' \
  --output-dir results/YOUR_RUN/metrics_3d --percentile 95

FP/FN are slice-class presence errors, not pixel counts. HD is computed only when both masks contain the class.
2D outputs have shape (slices, classes).
3D outputs have shape (patients, classes).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from PIL import Image
from monai.metrics import compute_hausdorff_distance
from monai.metrics import compute_dice


def read_labels(path, classes):
    """
    Decode the saved grayscale PNG into class IDs (0, 1, 2, ...).
    """
    with Image.open(path) as image:
        values = np.asarray(image)
    assert values.ndim == 2, f'{path}: expected a grayscale image'

    # SEGTHOR saves classes 0..4 as pixel values 0, 63, 126, 189, 252
    if classes == 5:
        step = 63
    else:
        step = 255 / (classes - 1)

    labels = np.full(values.shape, -1, dtype=int)

    for class_id in range(classes):
        pixel_value = int(class_id * step)
        labels[values == pixel_value] = class_id

    assert np.all(labels >= 0), f'{path}: unexpected label values'
    return labels


def evaluate_hd_in_2d(args):
    """
    Evaluate Hausdorff distance and presence errors for a single prediction directory.
    """
    files = sorted(args.pred_dir.glob('*.png'))  # use filename order consistently across all output arrays
    assert files, f'No prediction PNGs found in {args.pred_dir}'

    with args.spacing_file.open('rb') as stream:
        spacings = pickle.load(stream)

    shape = (len(files), args.classes) # rows are slices, columns are classes
    hd_values = np.full(shape, np.nan, dtype=np.float32)
    false_positives = np.zeros(shape, dtype=bool)
    false_negatives = np.zeros(shape, dtype=bool)
    slice_names = []

    for slice_index, path in enumerate(files):
        pred = read_labels(path, args.classes)
        gt = read_labels(args.gt_dir / path.name, args.classes)
        slice_names.append(path.stem)
        assert pred.shape == gt.shape, f'{path.name}: prediction and ground truth sizes differ'

        # spacing must describe the resized slice, in (row, column) order
        patient = path.stem.rsplit('_', 1)[0]
        spacing = np.asarray(spacings[patient][:2], dtype=float)

        assert spacing.shape == (2,) and np.all(np.isfinite(spacing) & (spacing > 0)), f'{patient}: invalid in-plane spacing {spacing}'

        for k in range(1, args.classes):  # skip background
            prediction = pred == k
            target = gt == k
            pred_present = prediction.any()
            gt_present = target.any()

            # count presence errors per slice and organ, not pixels
            if pred_present and not gt_present:
                false_positives[slice_index, k] = True
                continue
            if gt_present and not pred_present:
                false_negatives[slice_index, k] = True
                continue
            if not pred_present and not gt_present:
                continue

            # add batch and channel dimensions for MONAI 
            prediction_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
            
            hd = compute_hausdorff_distance(
                prediction_tensor,
                target_tensor,
                include_background=True,    # the single channel is an organ, so keep that organ
                percentile=args.percentile,
                directed=False,
                distance_metric='euclidean',
                spacing=spacing.tolist()).item()
                
            assert np.isfinite(hd), f'{path.name}, class {k}: nonfinite HD for nonempty masks'

            hd_values[slice_index, k] = hd

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'hd_val.npy', hd_values) 
    np.save(args.output_dir / 'fp_val.npy', false_positives) # summing over axis 0 gives per-class counts
    np.save(args.output_dir / 'fn_val.npy', false_negatives) # summing over axis 0 gives per-class counts
    np.save(args.output_dir / 'stems_val.npy', np.asarray(slice_names)) # records which slice each row belongs to

    # Print summary
    for k in range(1, args.classes):
        distances = hd_values[:, k]
        valid_count = np.isfinite(distances).sum()
        if valid_count > 0:
            mean_hd = np.nanmean(distances)
            std_hd = np.nanstd(distances)  
        else: 
            mean_hd = float('nan') # classes without valid pairs have undefined mean HD
            std_hd = float('nan')
        fp_count = false_positives[:, k].sum()
        fn_count = false_negatives[:, k].sum()
        gt_present_count = valid_count + fn_count
        print(f'Class {k}: HD{args.percentile:g} mean={mean_hd:.3f} mm, std={std_hd:.3f} mm, '
              f'total={len(files)}, valid={valid_count}, invalid={len(files) - valid_count}, '
              f'FP={fp_count}, FN={fn_count}, '
              f'GT present={gt_present_count}, GT absent={len(files) - gt_present_count}')
              
    print(f'Total slices: {len(files)}; foreground slice-class pairs: {len(files) * (args.classes - 1)}')


def evaluate_dice_and_hd_in_3d(args):
    files = sorted(args.pred_dir.glob('*.nii.gz'))
    assert files, f'No stitched NIfTI files found in {args.pred_dir}'

    patient_names = []
    shape = (len(files), args.classes)
    dice_values = np.full(shape, np.nan, dtype=np.float32)
    hd_values = np.full(shape, np.nan, dtype=np.float32)

    for patient_index, path in enumerate(files):
        patient = path.name.removesuffix('.nii.gz')
        patient_names.append(patient)
        gt_filename = args.gt_pattern.format(patient=patient)
        gt_path = args.gt_dir / gt_filename

        # load patient's prediction and matching ground truth
        pred_image = nib.load(path)
        gt_image = nib.load(gt_path)
        pred = np.asarray(pred_image.dataobj)
        gt = np.asarray(gt_image.dataobj)

        # CHECKS
        assert pred.ndim == 3, f'{patient}: prediction must be 3D'
        assert pred.shape == gt.shape, f'{patient}: prediction and GT shapes differ'
        assert np.allclose(pred_image.affine, gt_image.affine, rtol=0, atol=1e-4), f'{patient}: prediction and GT are on different physical grids'
        assert np.isin(pred, range(args.classes)).all(), f'{patient}: unexpected prediction labels'
        assert np.isin(gt, range(args.classes)).all(), f'{patient}: unexpected GT labels'

        # get voxel spacing
        spacing = []
        for value in pred_image.header.get_zooms():
            assert np.isfinite(value) and value > 0, f'{patient}: invalid voxel spacing'
            spacing.append(float(value))
        assert len(spacing) == 3, f'{patient}: expected three spacing values'

        for k in range(1, args.classes):  # skip background
            prediction = pred == k
            target = gt == k
            pred_present = prediction.any()
            gt_present = target.any()
        
            if not pred_present and not gt_present:
                print(f"Warning: Both prediction and ground truth are empty for patient {patient}, class {k}. Skipping Dice and HD computation.")
                continue

            # Add batch and channel dimensions for MONAI
            prediction_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)
            dice = compute_dice(
                prediction_tensor,
                target_tensor,
                include_background=True,  # channel contains one organ
                ignore_empty=False).item()
            
            dice_values[patient_index, k] = dice

            # HD needs both surfaces - leave it as NaN if either is empty
            if pred_present and gt_present:
                hd = compute_hausdorff_distance(
                    prediction_tensor,
                    target_tensor,
                    include_background=True, # channel contains one organ
                    percentile=args.percentile,
                    directed=False,
                    distance_metric='euclidean',
                    spacing=spacing).item()
           
                hd_values[patient_index, k] = hd

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'dice_3d_val.npy', dice_values)
    np.save(args.output_dir / 'hd_3d_val.npy', hd_values)
    np.save(args.output_dir / 'patients_val.npy', np.asarray(patient_names))

    # Collect defined scores for each organ, then average across patients
    for k in range(1, args.classes): # skip background
        valid_hd = []
        valid_dice = []
        for patient_index in range(len(patient_names)):
            hd = hd_values[patient_index, k]
            dice = dice_values[patient_index, k]
            if np.isfinite(hd):
                valid_hd.append(hd)
            if np.isfinite(dice):
                valid_dice.append(dice)

        mean_hd = float('nan')
        mean_dice = float('nan')
        std_hd = float('nan')
        std_dice = float('nan')

        if len(valid_hd) > 0:
            mean_hd = np.mean(valid_hd)
            std_hd = np.std(valid_hd)  
        if len(valid_dice) > 0:
            mean_dice = np.mean(valid_dice)
            std_dice = np.std(valid_dice)

        total = len(patient_names)
        print(f'Class {k}: total patients={total}')
        print(f'  Dice: mean={mean_dice:.3f}, std={std_dice:.3f}, '
              f'valid={len(valid_dice)}, invalid={total - len(valid_dice)}')
        print(f'  HD{args.percentile:g}: mean={mean_hd:.3f} mm, std={std_hd:.3f} mm, '
              f'valid={len(valid_hd)}, invalid={total - len(valid_hd)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred-dir', type=Path, required=True,
                        help='2D: saved PNG directory. 3D: stitched Patient_XX.nii.gz directory')
    parser.add_argument('--gt-dir', type=Path, required=True)
    parser.add_argument('--gt-pattern', default='{patient}.nii.gz',
                        help='3D GT path relative to gt-dir')
    parser.add_argument('--spacing-file', type=Path,
                        help='Required for 2D only: spacing.pkl with corrected resized spacing')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--percentile', type=float, choices=[50, 90, 95, 100], default=95)
    parser.add_argument('--dimensionality', type=int, choices=[2, 3], default=3)
    args = parser.parse_args()

    assert 2 <= args.classes <= 256, 'classes must be 2..256'

    if args.dimensionality == 2:
        assert args.spacing_file is not None, '--spacing-file is required for 2D evaluation'
        evaluate_hd_in_2d(args)
    elif args.dimensionality == 3:
        evaluate_dice_and_hd_in_3d(args)
