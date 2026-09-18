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
import pickle
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
                   monai_percentile_hausdorff_distance,
                   save_images)

from losses import (CrossEntropy)
import random

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# Added
datasets_params["SEGTHOR_FINAL"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_clip"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_bspline"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_FINAL_clip_bspline"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

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
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
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


    if args.bspline_slices:
        exclude_set = BAD_SLICES_BSPLINE 
    else:
        exclude_set = BAD_SLICES

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
                              num_workers=5,
                              shuffle=True,
                              generator=generator)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug, exclude=exclude_set) #added
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False,
                            generator=generator)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)
    patient_spacing = None
    if args.dataset != 'TOY2':
        # spacing.pkl contains resized slice (row, column, through-plane) spacing
        with open(Path('data') / args.dataset / 'spacing.pkl', 'rb') as f:
            patient_spacing = pickle.load(f)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))

    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))
    log_hd_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

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
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class
                    
                    if m == 'val':
                        spacing = None if patient_spacing is None else [
                            [float(s) for s in patient_spacing[stem.rsplit('_', 1)[0]][:2]]
                            for stem in data['stems']]
                        log_hd_val[e, j:j + B, :] = monai_percentile_hausdorff_distance(
                            pred_seg, gt, percentile=args.hd_percentile, spacing=spacing).detach().cpu()

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
                    if m == 'val':
                        postfix_dict[f"HD{args.hd_percentile}"] = f"{log_hd_val[e, :j, 1:].nanmean():05.3f}"
                        if K > 2:
                            postfix_dict |= {f"HD{args.hd_percentile}-{k}": f"{log_hd_val[e, :j, k].nanmean():05.3f}"
                                             for k in range(1, K)}
                                 
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        np.save(args.dest / "hd_val.npy", log_hd_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            current_hd = log_hd_val[e, :, 1:].nanmean().item()
            message = (f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
                       f" | Validation HD{args.hd_percentile}: {current_hd:.3f}")
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--bspline_slices', action='store_true',
                     help="Use the bad-slice list for B-spline-resampled datasets")
    parser.add_argument('--hd_percentile', type=int, default=95, choices=[90, 95, 99, 100], help="Percentile for Hausdorff distance computation")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    pprint(args)
    # Set seed
    set_seed(args.seed)
    runTraining(args)


if __name__ == '__main__':
    main()
