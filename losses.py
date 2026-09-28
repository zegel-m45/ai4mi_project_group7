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


from torch import einsum
from monai.losses import DiceLoss, GeneralizedDiceLoss
from utils import simplex, sset

class CombinedLoss():
    def __init__(self, idk, alpha=0.5, generalized=False):
        self.idk = idk
        self.alpha = alpha # scalar for which loss gets more weight
        self.ce = CrossEntropy(idk=idk)
        dice_loss = GeneralizedDiceLoss if generalized else DiceLoss
        # In main.py we already apply F.softmax (so therefore False here)
        self.dice = dice_loss(include_background=True, softmax=False, reduction="mean")
        # ^ The GeneralizedDiceLoss has batch=False by default. This is important because now the volume is
        # computed per slice! If we set batch=True, it will take the volume of the batch I believe, 
        # but the batch during training can be multiple patients (training dataloader has shuffle=True).
        # Thus that volume does not make sense I think to use.

        print(f"Initialized {self.__class__.__name__} with idk={idk}, alpha={alpha}, generalized={generalized}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        ce_loss = self.ce(pred_softmax, weak_target)

        pred = pred_softmax[:, self.idk, ...]
        gt = weak_target[:, self.idk, ...].float()
        dice_loss = self.dice(pred, gt)

        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
