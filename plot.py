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

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def run(args: argparse.Namespace) -> None:
    metrics: np.ndarray = np.load(args.metric_file)
    # Exclude both-empty masks, while retaining finite one-empty penalties
    is_hd = args.metric_file.stem.startswith('hd')
    average = np.nanmean if is_hd else np.mean
    match metrics.ndim:
        case 2:
            E, N = metrics.shape
            K = 1
        case 3:
            E, N, K = metrics.shape

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(str(args.metric_file))

    epcs = np.arange(E)

    if metrics.ndim == 3:
        for k in range(1, K):
            y = average(metrics[:, :, k], axis=1)
            ax.plot(epcs, y, label=f"{k=}", linewidth=1.5)
        if K > 1:
            ax.plot(epcs, average(metrics[:, :, 1:], axis=(1, 2)),
                    label="Foreground mean", linewidth=3)  
            if not is_hd:
                ax.plot(epcs, average(metrics, axis=(1, 2)), label="All classes", linewidth=3)
        else:
            ax.plot(epcs, average(metrics[:, :, 0], axis=1), label="k=0", linewidth=3)
        ax.legend()
    else:
        ax.plot(epcs, average(metrics, axis=1), linewidth=3)
    ax.set_xlabel("Epoch")

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--metric_file', type=Path, required=True, metavar="METRIC_MODE.npy",
                        help="The metric file to plot.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided.")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
