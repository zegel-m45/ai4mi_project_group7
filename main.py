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
import json
import random
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

from losses import CrossEntropy, MarginalCrossEntropy
from segthor_metrics import segthor_dice, segthor_coarse_probs, CoarsePatientDice

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# Added
datasets_params["SEGTHOR_clip"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_bspline_flipped"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
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

def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


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
    B: int = args.batch_size or datasets_params[args.dataset]['B']
    root_dir = args.data_dir or Path("data") / args.dataset
    loader_options = {}
    if args.seed is not None:
        loader_options = {'worker_init_fn': seed_worker,
                          'generator': torch.Generator().manual_seed(args.seed)}



    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug,
                             require_annotations=args.loss == 'marginal')
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              **loader_options,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug,
                           require_annotations=args.loss == 'marginal')
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            **loader_options,
                            shuffle=False)

    if train_set.has_annotations:
        if K != 5:
            raise ValueError("SegTHOR annotation metadata requires five output classes.")
        overlap = set(train_set.patient_ids) & set(val_set.patient_ids)
        if overlap:
            raise ValueError(f"Patients occur in both training and validation: {sorted(overlap)}")
        if args.dest.exists() and any(args.dest.iterdir()):
            raise ValueError(f"Results folder is not empty: {args.dest}. Choose a new --dest.")
    if args.loss == 'marginal' and not train_set.fine_patients:
        raise ValueError("Marginal training requires Patient 7 GT2 in the training data.")
    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    if args.loss == 'marginal' and (args.mode != 'full' or not args.dataset.startswith('SEGTHOR')):
        raise ValueError("--loss marginal requires a SegTHOR dataset and --mode full.")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.loss == 'marginal':
        loss_fn = MarginalCrossEntropy()
    elif args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    annotated = train_loader.dataset.has_annotations
    if annotated:
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update(device=str(device), torch_version=torch.__version__, numpy_version=np.__version__,
                      resolved_data_dir=str(train_loader.dataset.root_dir),
                      effective_batch_size=train_loader.batch_size,
                      checkpoint_metric='mean patient Dice: union, heart, trachea')
        config['annotation_manifest'] = json.loads(
            (Path(train_loader.dataset.root_dir) / 'annotations.json').read_text())
        config['coarse_class_order'] = ['background', 'esophagus_or_aorta', 'heart', 'trachea']
        config['fine_class_order'] = ['background', 'esophagus', 'heart', 'trachea', 'aorta']
        config['dice_unavailable'] = 'NaN for esophagus and aorta on merged-label cases'
        config['coarse_prediction_rule'] = 'sum probabilities of classes 1 and 4 before argmax'
        config['prediction_encoding'] = {'val': 'fine labels 0..4 multiplied by 63',
                                         'val_coarse': 'coarse labels 0..3 multiplied by 63'}
        config['patient_order'] = {name: loader.dataset.patient_ids
                                   for name, loader in [('tra', train_loader), ('val', val_loader)]}
        config['debug_metrics'] = 'Only selected slices are scored in --debug; these are not full volumes.'
        (args.dest / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    coarse_logs = {}
    patient_logs = {}
    if annotated:
        for name, loader in [('tra', train_loader), ('val', val_loader)]:
            coarse_logs[name] = torch.full((args.epochs, len(loader.dataset), 4), float('nan'))
            patient_logs[name] = torch.full((args.epochs, len(loader.dataset.patient_ids), 4), float('nan'))
        log_dice_tra.fill_(float('nan'))
        log_dice_val.fill_(float('nan'))

    best_dice: float = -float('inf') if annotated else 0

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
                split_name = 'tra' if m == 'train' else 'val'
                patient_meter = CoarsePatientDice(loader.dataset.patient_ids) if annotated else None
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    if annotated:
                        with torch.no_grad():
                            is_fine = data['is_fine'].to(device)
                            fine_dice, coarse_dice = segthor_dice(pred_probs, gt, is_fine)
                            log_dice[e, j:j + B, :] = fine_dice.cpu()
                            coarse_logs[split_name][e, j:j + B, :] = coarse_dice.cpu()
                            patient_meter.update(pred_probs, gt, data['patient_ids'])
                    else:
                        pred_seg = probs2one_hot(pred_probs)
                        log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = (loss_fn(pred_logits, gt, is_fine) if args.loss == 'marginal'
                            else loss_fn(pred_probs, gt))
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
                            if annotated:
                                coarse_class = probs2class(segthor_coarse_probs(pred_probs))
                                save_images(coarse_class * 63, data['stems'],
                                            args.dest / f"iter{e:03d}" / 'val_coarse')

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if annotated:
                        postfix_dict = {"Coarse-slice-Dice": f"{coarse_logs[split_name][e, :j, 1:].mean():05.3f}",
                                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2 and not annotated:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

                if annotated:
                    patient_logs[split_name][e] = patient_meter.compute()

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        if annotated:
            for name in ('tra', 'val'):
                np.save(args.dest / f'dice_coarse_{name}.npy', coarse_logs[name])
                np.save(args.dest / f'dice_coarse_patient_{name}.npy', patient_logs[name])
            current_dice = patient_logs['val'][e, :, 1:].mean().item()
        else:
            current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            if annotated:
                previous = "initial" if best_dice == -float("inf") else f"{best_dice:05.3f}"
                message = f">>> Improved mean patient coarse Dice at epoch {e}: {previous}->{current_dice:05.3f} DSC"
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
    parser.add_argument('--loss', choices=['ce', 'marginal'], default='ce')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--num-workers', type=int, default=5)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")

    args = parser.parse_args()
    if args.num_workers < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('num-workers must be nonnegative and batch-size must be positive')
    if args.seed is not None and not 0 <= args.seed < 2 ** 32:
        parser.error('seed must be between 0 and 2**32 - 1')

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
