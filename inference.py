import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ENet import ENet
from main import img_transform
from stitch import main as stitch_predictions
from utils import save_images

# Final model settings
DCA = True
FACTOR = 2
INPUT_SLICES = 13
CLASSES = 5
KERNELS = 16
BATCH_SIZE = 8 # Does not matter for inference, only memory
GROUP_REGEX = r"(Patient_\d+)_\d+$"


def load_model(args, device):
    model_class = ENet

    model = model_class(
        in_dim=INPUT_SLICES,
        out_dim=CLASSES,
        kernels=KERNELS,
        factor=FACTOR,
        use_dca=DCA,
    ).to(device)

    weights = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(weights, dict) and 'state_dict' in weights:
        weights = weights['state_dict']
   
    if weights and all(key.startswith('module.') for key in weights):
        weights = {key.removeprefix('module.'): value for key, value in weights.items()}

    model.load_state_dict(weights)
    model.eval()

    return model


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dir = args.image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image dir does not exist: {image_dir}')

    dataset = SliceDataset(
        'test',
        image_dir.parent.parent,
        img_transform=img_transform,
        context_slices=INPUT_SLICES,
    )
    if len(dataset) == 0:
        raise ValueError(f'Empty dataset: image dir: {image_dir}')

    model = load_model(args, device)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
    )

    iteration_dir = args.output_dir / 'iter000'
    slice_output = iteration_dir / 'test'
    volume_output = iteration_dir / 'test_volumes'

    with torch.inference_mode():
        for batch in loader:
            images = batch['images'].to(device, non_blocking=device.type == 'cuda')
            predicted_class = model(images).argmax(dim=1)
            save_images(predicted_class * 63, batch['stems'], slice_output)

    stitch_predictions(argparse.Namespace(
        data_folder=slice_output,
        dest_folder=volume_output,
        grp_regex=GROUP_REGEX,
        num_classes=255,
        source_scan_pattern=args.source_scan_pattern,
        bspline=args.bspline,
        new_spacing=args.new_spacing,
    ))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Inference on slices (from test set)'
    )
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to the training run weights (bestweights.pt)')
    parser.add_argument('--image-dir', type=Path,
                        default=Path('/gpfs/home3/scur0178/ai4mi_project_group7/data/SEGTHOR_test_renamed/test/img'),
                        help='Directory containing NPY slices named Patient_XX_####.npy.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory for the stitched .nii.gz predictions.')
    parser.add_argument('--source-scan-pattern',
                        default='/gpfs/home3/scur0178/ai4mi_project_group7/data/segthor_full/test/{id_}.nii.gz',
                        help='Used for stitching the data.')
    parser.add_argument('--num-workers', type=int, default=5)

    # Necessary to use for reconstruction
    parser.add_argument('--bspline', action='store_true',
                        help='Restore predictions with B-spline resampling.')
    parser.add_argument('--new-spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Spacing used when preprocessing B-spline datasets.')
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    if '{id_}' not in args.source_scan_pattern:
        parser.error('--source-scan-pattern must contain the {id_} placeholder.')

    return args


if __name__ == '__main__':
    infer(parse_args())
