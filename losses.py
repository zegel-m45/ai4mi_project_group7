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


from typing import Any, Callable, List, Tuple
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as eucl_distance
from torch import Tensor, einsum
from monai.losses import DiceLoss, GeneralizedDiceLoss
from utils import simplex, sset, one_hot

class CombinedLoss():
    def __init__(self, idk, alpha=0.5, generalized=False, weight=None, boundary_weight=0.0):
        self.idk = idk
        self.alpha = alpha # scalar for which loss gets more weight
        # [weight of the CE/Dice loss combi, weight of the boundary loss term] at the start 
        # (StealWeight will gradually change it after each epoch)
        self.weights = [1.0, boundary_weight]
        self.boundary = SurfaceLoss(idc=idk) if boundary_weight > 0 else None
        self.ce = CrossEntropy(idk=idk, weight=weight)
        dice_loss = GeneralizedDiceLoss if generalized else DiceLoss
        # In main.py we already apply F.softmax (so therefore False here)
        self.dice = dice_loss(include_background=True, softmax=False, reduction="mean")
        # ^ The GeneralizedDiceLoss has batch=False by default. This is important because now the volume is
        # computed per slice! If we set batch=True, it will take the volume of the batch I believe, 
        # but the batch during training can be multiple patients (training dataloader has shuffle=True).
        # Thus that volume does not make sense I think to use.

        # For logging
        self.last_ce = 0.0
        self.last_dice = 0.0
        self.last_boundary = 0.0

        print(f"Initialized {self.__class__.__name__} with idk={idk}, alpha={alpha}, generalized={generalized}, "
              f"boundary_weight={boundary_weight}")

    def __call__(self, pred_softmax, weak_target, dist_maps=None):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        ce_loss = self.ce(pred_softmax, weak_target)

        pred = pred_softmax[:, self.idk, ...]
        gt = weak_target[:, self.idk, ...].float()
        dice_loss = self.dice(pred, gt)

        self.last_ce = ce_loss.item()
        self.last_dice = dice_loss.item()

        loss = self.weights[0] * (self.alpha * ce_loss + (1 - self.alpha) * dice_loss)

        if self.boundary is not None:
            assert dist_maps is not None, "boundary_weight > 0 needs the distance maps of the ground truth"
            boundary_loss = self.boundary(pred_softmax, dist_maps)
            self.last_boundary = boundary_loss.item()
            loss = loss + self.weights[1] * boundary_loss

        return loss


# Copied from: https://github.com/LIVIAETS/boundary-loss (scheduler.py)
class StealWeight():
    def __init__(self, to_steal: float):
        self.to_steal: float = to_steal

    def __call__(self, epoch: int, optimizer: Any, loss_fns: list[list[Callable]], loss_weights: list[list[float]]) \
            -> Tuple[float, list[list[Callable]], list[list[float]]]:
        new_weights: list[list[float]] = [[max(0.1, a - self.to_steal), b + self.to_steal] for a, b in loss_weights]

        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights


# Copied from: https://github.com/LIVIAETS/boundary-loss
class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss




class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        self.weight = kwargs.get('weight', None)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1]) # ground truth is one-hot -> use for weights

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        if self.weight is not None:
            # make dim 4D (to match mask [b,k,w,h]): [1, k, 1, 1]
            w = self.weight[self.idk].to(mask.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mask = mask * w

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
