#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
from html import parser
import math
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy, CombinedLoss)
import random
from stitch import main as stitch_predictions
from evaluate_metrics_offline import evaluate_dice_and_hd_in_3d

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_baseline"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# Added
datasets_params["SEGTHOR_FINAL"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_clip_bspline_zscore_patient"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_clip"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_bspline"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_clip_bspline"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_minmax_dataset"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_zscore_dataset"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_zscore_patient"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_clip_zscore_patient_bspline"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

# Source - https://stackoverflow.com/a/64584503 
# Posted by yeachan park, modified by community. See post 'Timeline' for change history 
# Retrieved 2026-09-18, License - CC BY-SA 4.0
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def img_transform(img):
        if isinstance(img, np.ndarray):
            return torch.as_tensor(img[np.newaxis, ...], dtype=torch.float32) # for images produced by z-scoring
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = args.kernels
    factor: int = args.factor
    net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 betas=(args.beta1, args.beta2))

    # Dataset part
    B: int = args.batch_size
    root_dir = Path("data") / args.dataset

    BAD_SLICES = {
        "Patient_03_0080",
        "Patient_03_0081",
        "Patient_03_0082",
        "Patient_04_0050",
        "Patient_04_0060",
        "Patient_05_0127",
        "Patient_05_0128",
        "Patient_05_0129",
        "Patient_05_0130",
        "Patient_13_0042",
        "Patient_14_0091",
        "Patient_15_0058",
        "Patient_15_0059",
        "Patient_15_0090",
        "Patient_15_0092",
        "Patient_17_0095",
        "Patient_18_0116",
        "Patient_19_0062",
        "Patient_19_0104",
        "Patient_19_0105"
    }

    BAD_SLICES_BSPLINE = {
        "Patient_03_0199", "Patient_03_0200", "Patient_03_0201", "Patient_03_0202", "Patient_03_0203", "Patient_03_0204", "Patient_03_0205", "Patient_03_0206", 
        "Patient_04_0124", "Patient_04_0125", "Patient_04_0126", "Patient_04_0149", "Patient_04_0150", "Patient_04_0151",
        "Patient_05_0254", "Patient_05_0255", "Patient_05_0256", "Patient_05_0257", "Patient_05_0258", "Patient_05_0259", "Patient_05_0260",
        "Patient_13_0083", "Patient_13_0084", 
        "Patient_14_0227", "Patient_14_0228", 
        "Patient_15_0144", "Patient_15_0145", "Patient_15_0146", "Patient_15_0147", "Patient_15_0148", 
        "Patient_15_0224", "Patient_15_0225", "Patient_15_0226", "Patient_15_0229", "Patient_15_0230", "Patient_15_0231", 
        "Patient_17_0237", "Patient_17_0238", 
        "Patient_18_0289", "Patient_18_0290", "Patient_18_0291", 
        "Patient_19_0154", "Patient_19_0155", "Patient_19_0156", 
        "Patient_19_0259", "Patient_19_0260", "Patient_19_0261", "Patient_19_0262",  "Patient_19_0263"
        }


    # Legacy slice exclusions are disabled for the corrected dataset.
    # To re-enable them, uncomment the if/else block below. Use
    # --bspline_slices only for the B-spline-resampled slice numbering.
    exclude_set = None
    # if args.bspline_slices:
    #     exclude_set = BAD_SLICES_BSPLINE
    # else:
    #     exclude_set = BAD_SLICES

    # Set seed to Dataloader
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug, exclude=exclude_set) #added
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True,
                              generator=generator)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug, exclude=exclude_set) #added
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False,
                            generator=generator)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)
    use_3d = (args.dataset != 'TOY2')
    log_dice_3d, log_hd_3d = [], []
    validation_patients = None

    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    if args.loss == "ce":
        loss_fn = CrossEntropy(idk=idk)
    elif args.loss == "combined":
        loss_fn = CombinedLoss(idk=idk, alpha=args.dice_alpha, generalized=args.generalized_dice)
    else:
        raise ValueError(args.loss)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))

    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert torch.isfinite(img).all(), "Input contains non-finite intensities" # images produced by z-scoring are no longer 0 to 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(args.logit_scale * pred_logits, dim=1)

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class
                    
                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                                 
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()

        if use_3d:
            epoch_folder = args.dest / f"iter{e:03d}"
            dice_3d, hd_3d, patients = evaluate_validation_3d(args, epoch_folder, K)
            
            if validation_patients is not None and patients != validation_patients:
                raise ValueError('Validation patient order changed between epochs.')
            validation_patients = patients
            log_dice_3d.append(dice_3d)
            np.save(args.dest / 'dice_3d_val.npy', np.asarray(log_dice_3d))

            if args.calculate_val_3d_hd:
                log_hd_3d.append(hd_3d)
                np.save(args.dest / 'hd_3d_val.npy', np.asarray(log_hd_3d))

            np.save(args.dest / 'patients_val.npy', np.asarray(patients))
    
            # Average patients across each foreground organ, then average across organs
            current_dice = float(np.nanmean(np.nanmean(dice_3d[:, 1:], axis=0)))
            if not np.isfinite(current_dice):
                raise ValueError('No valid foreground 3D Dice for checkpoint selection.')
           
            # Delete this epoch's prediction PNGs and val folder after successful evaluation
            for path in (epoch_folder / 'val').glob('*.png'):
                path.unlink()
            if not any((epoch_folder / 'val').iterdir()):
                (epoch_folder / 'val').rmdir()

        if current_dice > best_dice:
            metric = "3D Dice" if use_3d else "2D Dice"
            message = (f">>> Improved {metric} at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def evaluate_validation_3d(args, epoch_folder, classes):
    
    # Reconstruct the 3D volumes from the 2D predictions
    volumes = epoch_folder / "val_volumes"
    stitch_predictions(argparse.Namespace(
        data_folder=epoch_folder / "val", dest_folder=volumes,
        grp_regex=r"(Patient_\d+)_\d+$", num_classes=255,
        source_scan_pattern=args.source_scan_pattern,
        bspline=args.bspline_slices or "bspline" in args.dataset.lower(),
        new_spacing=args.new_spacing))
    
    # Evaluate the 3D Dice and optionally the 3D Hausdorff distance
    results = evaluate_dice_and_hd_in_3d(argparse.Namespace(
        pred_dir=volumes, gt_dir=args.gt_3d_dir, gt_pattern=args.gt_3d_pattern,
        output_dir=epoch_folder / 'metrics_3d', classes=classes,
        dice_only=not args.calculate_val_3d_hd, percentile=args.hd_percentile,
        device='cuda' if args.gpu and torch.cuda.is_available() else 'cpu',
        include_penalty=args.include_penalty))
    
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    # Loss params
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--loss', default='ce', choices=['ce', 'combined'],
                        help="ce: plain cross-entropy. combined: CE + (Generalized) Dice.")
    parser.add_argument('--dice-alpha', '--dice_alpha', type=float, default=0.5,
                        help="Weight of CE vs Dice for combined loss")
    parser.add_argument('--generalized-dice', '--generalized_dice', action='store_true',
                        help="Use GeneralizedDiceLoss instead of DiceLoss")

    parser.add_argument('--bspline_slices', action='store_true',
                     help="Reconstruct validation using the B-spline grid. Automatic for datasets containing 'bspline'.")
    parser.add_argument('--calculate-val-3d-hd', '--calculate_val_3d_hd', action='store_true',
                        help='Also evaluate 3D HD each epoch. Default computes only 3D Dice.')
    parser.add_argument('--hd-percentile', '--hd_percentile', type=float, default=95,
                        choices=[50, 90, 95, 100])
    parser.add_argument('--include-penalty', action='store_true',
                        help='Use the scan diagonal for one-empty 3D HD pairs.')
    parser.add_argument('--source-scan-pattern', default='data/segthor_full/train/{id_}/{id_}.nii.gz')
    parser.add_argument('--gt-3d-dir', type=Path, default=Path('data/segthor_full/train'))
    parser.add_argument('--gt-3d-pattern', default='{patient}/GT.nii.gz')
    parser.add_argument('--new-spacing', type=float, nargs=3, default=[1., 1., 1.],
                        help='Spacing used when preprocessing B-spline datasets.')
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0005,
                        help="Adam learning rate.")
    parser.add_argument('--beta1', type=float, default=0.9,
                        help="Adam beta for the running average of gradients.")
    parser.add_argument('--beta2', type=float, default=0.999,
                        help="Adam beta for the running average of squared gradients.")
    parser.add_argument('--batch-size', '--batch_size', type=int, default=None,
                        help="Batch size for training and validation; None uses the dataset default "
                             "(TOY2: 2, SEGTHOR variants: 8).")
    parser.add_argument('--num-workers', '--num_workers', type=int, default=5,
                        help="Data-loader workers; 0 loads data in the main process.")
    parser.add_argument('--kernels', type=int, default=None,
                        help="ENet base channel count, not spatial kernel size; "
                             "None uses the dataset default (8). SEGTHOR only.")
    parser.add_argument('--factor', type=int, default=None,
                        help="ENet bottleneck channel-reduction factor; "
                             "None uses the dataset default (2). SEGTHOR only.")
    parser.add_argument('--logit-scale', '--logit_scale', type=float, default=1.0,
                        help="Positive multiplier applied to logits before softmax "
                             "during training and validation (inverse temperature).")
    args = parser.parse_args(argv)

    if args.debug and args.dataset != 'TOY2':
        parser.error('3D validation needs complete patients. --debug truncates the slices.')
    if args.calculate_val_3d_hd and args.dataset == 'TOY2':
        parser.error('--calculate-val-3d-hd requires a SEGTHOR dataset.')
    valid_spacing = [math.isfinite(s) and s > 0 for s in args.new_spacing]
    if not all(valid_spacing):
        parser.error('--new-spacing values must be finite and positive.')
    if args.dataset == 'TOY2' and (args.kernels is not None or args.factor is not None):
        parser.error('--kernels and --factor apply only to ENet (SEGTHOR datasets).')

    # Resolve dataset defaults before printing or using the hyperparameters.
    params = datasets_params[args.dataset]
    if args.batch_size is None:
        args.batch_size = params['B']
    if args.kernels is None:
        args.kernels = params.get('kernels', 8)
    if args.factor is None:
        args.factor = params.get('factor', 2)

    if args.epochs < 1:
        parser.error('--epochs must be at least 1.')
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1.')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative.')
    if not math.isfinite(args.lr) or args.lr < 0:
        parser.error('--lr must be finite and non-negative.')
    if not (0 <= args.beta1 < 1) or not (0 <= args.beta2 < 1):
        parser.error('--beta1 and --beta2 must each be in [0, 1).')
    if args.kernels < 2 or not (1 <= args.factor <= args.kernels):
        parser.error('--kernels must be at least 2 and --factor must be between 1 and --kernels.')
    if not math.isfinite(args.logit_scale) or args.logit_scale <= 0:
        parser.error('--logit-scale must be finite and greater than 0.')

    return args


def main():
    args = parse_args()

    pprint(args)
    # Set seed
    set_seed(args.seed)
    runTraining(args)


if __name__ == '__main__':
    main()
