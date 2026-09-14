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


import torch
from torch import einsum

from utils import simplex, sset


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


class MarginalCrossEntropy:

    def __call__(self, pred_logits, target, is_fine):
        if not isinstance(pred_logits, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("pred_logits and target must be tensors")
        if pred_logits.ndim != 4 or pred_logits.shape[1] != 5:
            raise ValueError("pred_logits must have shape [B, 5, H, W]")
        if any(size == 0 for size in pred_logits.shape):
            raise ValueError("pred_logits must have non-empty batch and spatial dimensions")
        if target.shape != pred_logits.shape:
            raise ValueError("target must have the same shape as pred_logits")
        if not pred_logits.is_floating_point():
            raise TypeError("pred_logits must have a floating-point dtype")
        if target.is_complex():
            raise TypeError("target must contain real one-hot labels")
        if not isinstance(is_fine, torch.Tensor) or is_fine.dtype != torch.bool:
            raise TypeError("is_fine must be a boolean tensor, one flag per sample")
        if is_fine.shape != (pred_logits.shape[0],):
            raise ValueError("is_fine must have shape [B]")
        if target.device != pred_logits.device or is_fine.device != pred_logits.device:
            raise ValueError("pred_logits, target, and is_fine must be on the same device")
        if not torch.isfinite(pred_logits).all():
            raise ValueError("pred_logits must be finite")
        if not ((target == 0) | (target == 1)).all() or not (target.sum(dim=1) == 1).all():
            raise ValueError("target must be one-hot: exactly one class per pixel")
        if (target[~is_fine, 4] != 0).any():
            raise ValueError("coarse samples cannot contain label 4; check annotation metadata")


        logits = pred_logits.float() if pred_logits.dtype in (torch.float16, torch.bfloat16) else pred_logits
        log_p = torch.log_softmax(logits, dim=1)
        labels = target.to(dtype=torch.long).argmax(dim=1)
        exact_log_p = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
        merged_log_p = torch.logsumexp(log_p[:, [1, 4]], dim=1)
        merged_pixels = (~is_fine[:, None, None]) & (labels == 1)

        return -torch.where(merged_pixels, merged_log_p, exact_log_p).mean()
