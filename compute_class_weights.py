import argparse
from pathlib import Path

import torch
from PIL import Image
from utils import tqdm_

from dataset import make_dataset
from main import gt_transform

NUMBER_OF_CLASSES = 5
C = 1.02 # from Enet paper

def compute_pixel_counts(root_dir: Path, subset: str) -> torch.Tensor:
    files = make_dataset(root_dir, subset)
    counts = torch.zeros(NUMBER_OF_CLASSES, dtype=torch.float64)

    for _, gt_path in tqdm_(files, desc="Counting pixels:"):
        gt_one_hot = gt_transform(K=NUMBER_OF_CLASSES, img=Image.open(gt_path))
        counts += gt_one_hot.sum(dim=(1, 2)) # sum over width and height, so only class dimension stays

    return counts  # per class count


def class_weights_from_counts(counts: torch.Tensor, weighting_method: str) -> torch.Tensor:
    freq = counts / counts.sum()

    if weighting_method == "inverse":
        weights = 1.0 / freq
    elif weighting_method == "median":
        weights = freq.median() / freq
    # From Enet paper and this article: a custom Enet class weighting formula 
    # https://medium.com/data-science/enet-a-deep-neural-architecture-for-real-time-semantic-segmentation-2baa59cf97e9
    elif weighting_method == "enet":
        weights = 1.0 / torch.log(C + freq)
    else:
        raise ValueError(weighting_method)

    return weights


def parse_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, help="Dataset folder name")
    parser.add_argument('--subset', default='train', choices=['train', 'val'])
    parser.add_argument('--weighting_method', default='median', choices=['inverse', 'median', 'enet'])
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path("data") / args.dataset

    counts = compute_pixel_counts(root_dir, args.subset)
    freq = counts / counts.sum()
    weights = class_weights_from_counts(counts, args.weighting_method)

    print(f"Pixel counts (per class): {counts.tolist()}")
    print(f"Class frequency: {[f'{f:.6f}' for f in freq.tolist()]}")
    print(f"Class weights: ({args.weighting_method}):{[f'{w:.4f}' for w in weights.tolist()]}")

if __name__ == '__main__':
    main()
