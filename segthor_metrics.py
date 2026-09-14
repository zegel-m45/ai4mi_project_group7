
from collections.abc import Iterable

import torch
from torch import Tensor

from utils import dice_coef, probs2one_hot


def _validate_inputs(pred_probs: Tensor, gt: Tensor) -> None:
    if pred_probs.ndim != 4 or pred_probs.shape[1] != 5:
        raise ValueError("Expected predictions with shape [B, 5, H, W]")
    if gt.shape != pred_probs.shape or gt.device != pred_probs.device:
        raise ValueError("Predictions and targets must have matching shapes and devices")
    if not pred_probs.is_floating_point():
        raise ValueError("Predictions must be floating-point class probabilities")
    if not torch.isfinite(pred_probs).all() or (pred_probs < 0).any():
        raise ValueError("Predictions must be finite, nonnegative probabilities")
    if not torch.allclose(pred_probs.sum(dim=1), torch.ones_like(pred_probs[:, 0])):
        raise ValueError("Predicted class probabilities must sum to one")
    if not ((gt == 0) | (gt == 1)).all() or not (gt.sum(dim=1) == 1).all():
        raise ValueError("Targets must be one-hot encoded")


def segthor_coarse_probs(pred_probs: Tensor) -> Tensor:
    return torch.stack((pred_probs[:, 0], pred_probs[:, 1] + pred_probs[:, 4],
                        pred_probs[:, 2], pred_probs[:, 3]), dim=1)


def _coarse_masks(pred_probs: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:

    coarse_probs = segthor_coarse_probs(pred_probs)
    coarse_gt = torch.stack((gt[:, 0], gt[:, 1] + gt[:, 4], gt[:, 2], gt[:, 3]), dim=1)
    return probs2one_hot(coarse_probs), coarse_gt.to(torch.int32)


@torch.no_grad()
def segthor_dice(pred_probs: Tensor, gt: Tensor, is_fine: Tensor) -> tuple[Tensor, Tensor]:
    _validate_inputs(pred_probs, gt)
    fine = torch.as_tensor(is_fine, device=gt.device)
    if fine.dtype != torch.bool or fine.shape != (gt.shape[0],):
        raise ValueError("is_fine must contain one boolean per sample")
    if gt[~fine, 4].any():
        raise ValueError("A coarse annotation cannot contain the separate aorta label 4")

    fine_dice = dice_coef(probs2one_hot(pred_probs), gt.to(torch.int32))
    fine_dice[~fine, 1] = float("nan")
    fine_dice[~fine, 4] = float("nan")
    coarse_pred, coarse_gt = _coarse_masks(pred_probs, gt)
    return fine_dice, dice_coef(coarse_pred, coarse_gt)


class CoarsePatientDice:

    def __init__(self, patient_ids: Iterable[str]):
        if isinstance(patient_ids, str):
            raise ValueError("patient_ids must be an iterable of patient identifiers")
        ids = list(patient_ids)
        if not ids or any(not isinstance(pid, str) or not pid for pid in ids):
            raise ValueError("Patient identifiers must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("Patient identifiers must be unique")
        self._patient_ids = sorted(ids)
        self._index = {pid: i for i, pid in enumerate(self._patient_ids)}

        self._counts = torch.zeros((len(ids), 3, 4), dtype=torch.int64)
        self._seen = torch.zeros(len(ids), dtype=torch.bool)

    @property
    def patient_ids(self) -> list[str]:
        return self._patient_ids.copy()

    @torch.no_grad()
    def update(self, pred_probs: Tensor, gt: Tensor, patient_ids: Iterable[str]) -> None:
        _validate_inputs(pred_probs, gt)
        if isinstance(patient_ids, str):
            raise ValueError("Provide one patient identifier per sample")
        ids = list(patient_ids)
        if len(ids) != pred_probs.shape[0]:
            raise ValueError("Provide one patient identifier per sample")
        if any(pid not in self._index for pid in ids):
            raise ValueError("A batch contains a patient not registered in this meter")
        pred, target = _coarse_masks(pred_probs, gt)
        counts = torch.stack(((pred & target).sum(dim=(2, 3)),
                              pred.sum(dim=(2, 3)), target.sum(dim=(2, 3))), dim=1).cpu()
        indices = torch.tensor([self._index[pid] for pid in ids], dtype=torch.int64)
        self._counts.index_add_(0, indices, counts)
        self._seen[indices] = True

    def compute(self) -> Tensor:
        if not self._seen.all():
            missing = [pid for pid, seen in zip(self._patient_ids, self._seen) if not seen]
            raise ValueError(f"No slices accumulated for patients: {missing}")
        intersection = self._counts[:, 0].to(torch.float32)
        denominator = (self._counts[:, 1] + self._counts[:, 2]).to(torch.float32)
        return (2 * intersection + 1e-8) / (denominator + 1e-8)
