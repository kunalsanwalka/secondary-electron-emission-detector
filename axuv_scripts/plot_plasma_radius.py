#!/usr/bin/env python3
"""
Plot the plasma radius vs. time for a given shot, as determined by the AXUV data.

The radius is the RMS radius stored under the `radius_cm` key in the AXUV NPZ
files written by axuv_data_output.py (see plot_axuv_npz.py for everything else
that lives in those files).

The radius is averaged down into 10 kHz windows to suppress the noise on the
native (~1 MHz) data rate. The native data is drawn underneath in the same
colour, so the effect of the averaging stays visible.

Examples
--------
  # Plot every diode array available for the shot:
  python plot_plasma_radius.py -s 260426037

  # Just one diode array, saved to a file:
  python plot_plasma_radius.py -s 260426037 -d DA3 -o radius.png

  # Average into 5 kHz windows instead:
  python plot_plasma_radius.py -s 260426037 -r 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def find_npz_files(shot, diode=None):
    """Return the NPZ files in the data directory holding a radius for this shot."""

    pattern = f'AXUV_results*{shot}*{diode}*.npz' if diode else f'AXUV_results*{shot}*.npz'

    files = []
    for path in DATA_DIR.glob(pattern):
        with np.load(path, allow_pickle=True) as data:
            if f's{shot}_radius_cm' in data.files:
                files.append(path)

    # Duplicates such as '... DA1 copy.npz' sort after the original they copy.
    return sorted(files, key=lambda p: ('copy' in p.stem, p.name))


def average_down(time_s, values, rate_khz):
    """Average `values` into windows of width 1/rate_khz, returning bin centres."""

    window_s = 1 / (rate_khz * 1e3)
    bins = np.arange(time_s[0], time_s[-1] + window_s, window_s)

    means, edges, _ = binned_statistic(time_s, values, statistic='mean', bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Empty bins come back as NaN, which would break the line.
    keep = np.isfinite(means)

    return centres[keep], means[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--shot', required=True, type=int,
                        help='shot number to plot.')
    parser.add_argument('-d', '--diode', default=None,
                        help='diode array to plot, e.g. DA1 (default: all of them).')
    parser.add_argument('-r', '--rate', default=10.0, type=float,
                        help='averaging rate in kHz (default: 10).')
    parser.add_argument('-o', '--output', default=None,
                        help='save the figure here instead of showing it.')
    args = parser.parse_args()

    files = find_npz_files(args.shot, args.diode)
    if not files:
        sys.exit(f'No AXUV NPZ file with radius data for shot {args.shot} in {DATA_DIR}')

    fig, ax = plt.subplots(figsize=(8, 4.5))

    plotted = set()
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            time_s = data[f's{args.shot}_time_s']
            radius_cm = data[f's{args.shot}_radius_cm']
            diode_id = str(data[f's{args.shot}_diode_id']).split('.')[-1]

        # The data directory can hold duplicates of the same diode (e.g. ' copy'
        # files), so only the first file found for each diode gets plotted.
        if diode_id in plotted:
            print(f'Skipping {path.name}: {diode_id} is already plotted.')
            continue
        plotted.add(diode_id)

        time_avg_s, radius_avg_cm = average_down(time_s, radius_cm, args.rate)

        # The averaged trace sets the colour; the native data goes underneath it.
        line, = ax.plot(time_avg_s * 1000, radius_avg_cm,
                        linewidth=1.2, zorder=3, label=diode_id)
        ax.plot(time_s * 1000, radius_cm,
                color=line.get_color(), linewidth=0.6, alpha=0.2, zorder=2)

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('RMS Radius [cm]')
    ax.set_title(f'AXUV Plasma Radius — Shot {args.shot}\n'
                 f'{args.rate:g} kHz window average (native rate shown faded)',
                 fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f'Saved: {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
