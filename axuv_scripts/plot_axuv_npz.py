#!/usr/bin/env python3
"""
Read and compare AXUV NPZ files produced by axuv_data_output.py.

Examples
--------
  # List shots and plottable keys inside the file:
  python plot_axuv_npz.py -f results.npz -l

  # Plot all default 1-D quantities for every shot in the file:
  python plot_axuv_npz.py -f results.npz

  # Compare two shots on specific 1-D keys:
  python plot_axuv_npz.py -f results.npz -s 251009049 251009050 -k centroid_cm radius_cm

  # Show 2-D space×time maps for one shot:
  python plot_axuv_npz.py -f results.npz -s 251009049 -k signal_smooth_nA emissivity_profile_mW_per_cm3

  # Mix 1-D and 2-D keys in one call:
  python plot_axuv_npz.py -f results.npz -s 251009049 251009050 -k centroid_cm signal_smooth_nA
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── axis labels derived from key names ────────────────────────────────────────
LABELS = {
    "centroid_cm":                          "Centroid [cm]",
    "radius_cm":                            "RMS Radius [cm]",
    "m1_polarization_ratio":                "m=1 Polarisation Ratio",
    "signal_raw_nA":                        "Raw Signal [nA]",
    "signal_padded_nA":                     "Edge-filled Signal [nA]",
    "signal_smooth_nA":                     "Smoothed Signal [nA]",
    "signal_sym_smooth_nA":                 "Symmetric Signal (smooth) [nA]",
    "signal_asym_smooth_nA":                "Asymmetric Signal (smooth) [nA]",
    "emissivity_profile_mW_per_cm3":        "Emissivity [mW cm⁻³]",
    "macro_stability_chi_and_time_cm2_s":   "Macro-stability χ [cm²]",
    "time_s":                               "Time [s]  (x-axis for all 1D/2D plots)",
}

# Keys whose data array is 2-D (space × time) → pcolormesh
KEYS_2D = {
    "signal_raw_nA",
    "signal_padded_nA",
    "signal_smooth_nA",
    "signal_sym_smooth_nA",
    "signal_asym_smooth_nA",
    "emissivity_profile_mW_per_cm3",
}

# Matching stored x-axis (space) key for each 2-D key
XKEY = {
    "signal_raw_nA":               "impact_b_raw_m",
    "signal_padded_nA":            "impact_b_padded_m",
    "signal_smooth_nA":            "impact_b_smooth_m",
    "signal_sym_smooth_nA":        "impact_b_smooth_m",
    "signal_asym_smooth_nA":       "impact_b_smooth_m",
    "emissivity_profile_mW_per_cm3": "impact_b_smooth_half_m",
}

# Chi key — stored as object array wrapping a dict; needs special handling
KEY_CHI = "macro_stability_chi_and_time_cm2_s"

# Keys that hold auxiliary / non-plottable metadata
_META_KEYS = {
    "shot", "sample_rate_MHz", "plasma_edge_m", "plasma_length_m",
    "diode_id", "diode_description", "view_chord_deg",
    "filter_transmission", "power_cal_for_signal_padded_nA_kw_per_nA",
    "macro_stability_chi_sample_rate_khz",
    "impact_b_raw_m", "impact_b_padded_m",
    "impact_b_smooth_m", "impact_b_smooth_half_m",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path):
    return np.load(path, allow_pickle=True)


def _shots_in_file(data):
    shots = set()
    for k in data.files:
        head = k.split("_", 1)[0]
        if head.startswith("s") and head[1:].isdigit():
            shots.add(int(head[1:]))
    return sorted(shots)


def _plottable_keys(data, shot):
    prefix = f"s{shot}_"
    keys = []
    for k in data.files:
        if k.startswith(prefix):
            base = k[len(prefix):]
            if base not in _META_KEYS:
                keys.append(base)
    return sorted(keys)


def _get(data, shot, key):
    return data[f"s{shot}_{key}"]


def _label(key):
    return LABELS.get(key, key.replace("_", " "))


def _time_ms(data, shot):
    return _get(data, shot, "time_s") * 1000   # seconds → ms


def _x_cm(data, shot, key):
    xkey = XKEY[key]
    xvec = _get(data, shot, xkey)
    if xvec.ndim == 2:          # x_padded is (space, time) — columns are identical
        xvec = xvec[:, 0]
    return xvec * 100           # m → cm


# ── list mode ─────────────────────────────────────────────────────────────────

def list_contents(path, data):
    shots = _shots_in_file(data)
    print(f"\nFile : {path}")
    print(f"Shots: {shots}\n")
    if not shots:
        return
    keys = _plottable_keys(data, shots[0])
    print(f"{'Key':<46}  {'Type':<18}  Label")
    print("-" * 90)
    for k in keys:
        if k == "time_s":
            kind = "shared x-axis"
        elif k == KEY_CHI:
            khz     = _get(data, shots[0], "macro_stability_chi_sample_rate_khz")
            khz_str = ",".join(str(int(f)) for f in khz)
            kind    = f"{len(khz)}× {{t,χ}} @ {khz_str} kHz"
        elif k in KEYS_2D:
            kind = "2D (space × time)"
        else:
            kind = "1D (time series)"
        print(f"  {k:<44}  {kind:<18}  {_label(k)}")
        if k == KEY_CHI:
            print(f"    └─ add --norm to plot log(e·χ) [unitless], normalised by 1/e mm²")
    print()
    # print metadata for first shot
    sn = shots[0]
    print(f"Metadata (shot {sn}):")
    for mk in ["diode_id", "diode_description", "plasma_edge_m",
               "filter_transmission", "sample_rate_MHz"]:
        try:
            print(f"  {mk}: {_get(data, sn, mk)}")
        except KeyError:
            pass
    print()


# ── 1-D plot: all selected shots overlaid ─────────────────────────────────────

def _plot_1d(ax, data, shots, key, colors):
    for sn, col in zip(shots, colors):
        t   = _time_ms(data, sn)
        y   = _get(data, sn, key)
        ax.plot(t, y, color=col, label=f"Shot {sn}", linewidth=0.9)
    ax.set_ylabel(_label(key), fontsize=9)
    ax.set_title(_label(key), fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Time [ms]", fontsize=9)


# ── chi plot: each frequency window as a separate line ───────────────────────

def _plot_chi(ax, data, shots, colors, select_khz=None, norm=False):
    linestyles = ["-", "--", ":", "-."]
    # collect which windows will actually be plotted
    khz_list_ref = _get(data, shots[0], "macro_stability_chi_sample_rate_khz")
    active = [khz for khz in khz_list_ref
              if select_khz is None or float(khz) in select_khz]
    multi = len(active) > 1

    for sn, col in zip(shots, colors):
        chi_dict = _get(data, sn, KEY_CHI).item()
        khz_list = _get(data, sn, "macro_stability_chi_sample_rate_khz")
        li = 0
        for jjh, khz in enumerate(khz_list):
            if select_khz is not None and float(khz) not in select_khz:
                continue
            t   = chi_dict[f"{jjh}_t"] * 1000
            dev = chi_dict[f"{jjh}_dev"]
            if norm:
                dev = np.log(dev * 100) + 1
            ls  = linestyles[li % len(linestyles)] if multi else "-"
            ax.plot(t, dev, color=col, linestyle=ls, linewidth=0.9,
                    label=f"Shot {sn}  {khz} kHz")
            li += 1
    ylabel = "log(e·χ) [unitless]" if norm else _label(KEY_CHI)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(ylabel, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Time [ms]", fontsize=9)


# ── 2-D plot: one pcolormesh per shot ─────────────────────────────────────────

def _plot_2d(ax, data, shot, key, colorbar=True):
    t_ms  = _time_ms(data, shot)
    x_cm  = _x_cm(data, shot, key)
    z     = np.array(_get(data, shot, key), dtype=float)  # (space, time)

    # Guard against shape mismatch between stored x and z rows
    n_rows = z.shape[0]
    x_cm   = x_cm[:n_rows]

    # Thin time axis if very dense (keeps pcolormesh fast)
    if z.shape[1] > 3000:
        step  = z.shape[1] // 2000
        z     = z[:, ::step]
        t_ms  = t_ms[::step]

    pc = ax.pcolormesh(t_ms, x_cm, z, shading="auto", cmap="inferno")
    if colorbar:
        plt.colorbar(pc, ax=ax, label=_label(key), pad=0.02)
    ax.set_xlabel("Time [ms]", fontsize=9)
    ax.set_ylabel(
        f"{XKEY[key].replace('_', ' ')} [cm]", fontsize=9
    )
    ax.set_title(f"{_label(key)} — Shot {shot}", fontsize=10)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot AXUV NPZ output from axuv_data_output.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-f", "--file", required=True,
                        help="NPZ file path")
    parser.add_argument("-s", "--shots", nargs="+", type=int, default=None,
                        help="Shot numbers to compare (default: all shots in file)")
    parser.add_argument("-k", "--keys", nargs="+", default=None,
                        help="Data keys to plot (default: all 1-D keys). "
                             "Use -l to see available keys.")
    parser.add_argument("-n", "--norm", action="store_true",
                        help="Normalise chi: plot log(e·χ) = ln(χ_cm² · 100) + 1 [unitless]")
    parser.add_argument("-o", "--output", default=None,
                        help="Save figures to file instead of showing (e.g. -o fig.png). "
                             "Multiple figures get a numeric suffix: fig_1.png, fig_2.png ...")
    parser.add_argument("-hz", "--chi_khz", nargs="+", type=float, default=None,
                        help="Chi frequency window(s) to plot in kHz (e.g. -hz 20  or  -hz 20 1). "
                             "Default: all windows.")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List shots and keys then exit")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    data = _load(path)

    if args.list:
        list_contents(path, data)
        return

    # ── resolve shots ──────────────────────────────────────────────────────────
    all_shots = _shots_in_file(data)
    if not all_shots:
        sys.exit("No shots found in file.")
    shots = args.shots or all_shots
    missing = [s for s in shots if s not in all_shots]
    if missing:
        sys.exit(f"Shots not in file: {missing}\nAvailable: {all_shots}")

    # ── resolve keys ───────────────────────────────────────────────────────────
    all_keys = _plottable_keys(data, shots[0])
    if args.keys:
        bad = [k for k in args.keys if k not in all_keys]
        if bad:
            list_contents(path, data)
            sys.exit(f"Unknown key(s): {bad}")
        keys = args.keys
    else:
        keys = [k for k in all_keys if k not in KEYS_2D and k != "time_s"]  # default: 1-D + chi

    keys_1d  = [k for k in keys if k not in KEYS_2D and k != KEY_CHI]
    keys_2d  = [k for k in keys if k in  KEYS_2D]
    plot_chi = KEY_CHI in keys

    # colour cycle — one colour per shot for 1-D overlays
    cmap   = plt.colormaps["tab10"].resampled(max(len(shots), 1))
    colors = [cmap(i) for i in range(len(shots))]

    # ── 1-D figure: one row per key, shots overlaid ───────────────────────────
    all_1d_rows = keys_1d + ([KEY_CHI] if plot_chi else [])
    if all_1d_rows:
        n_rows = len(all_1d_rows)
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(10, 3.2 * n_rows),
                                 sharex=False)
        if n_rows == 1:
            axes = [axes]
        fig.suptitle(path.name, fontsize=11)
        for ax, key in zip(axes, keys_1d):
            _plot_1d(ax, data, shots, key, colors)
        if plot_chi:
            select_khz = set(args.chi_khz) if args.chi_khz else None
            _plot_chi(axes[-1], data, shots, colors, select_khz=select_khz, norm=args.norm)
        fig.tight_layout()

    # ── 2-D figures: one figure per key, one column per shot ─────────────────
    for key in keys_2d:
        n_cols = len(shots)
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(6 * n_cols, 5),
                                 sharey=True)
        if n_cols == 1:
            axes = [axes]
        fig.suptitle(f"{_label(key)}  |  {path.name}", fontsize=11)
        for ax, sn in zip(axes, shots):
            _plot_2d(ax, data, sn, key, colorbar=True)
        fig.tight_layout()

    if args.output:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        stem = Path(args.output)
        if len(figs) == 1:
            figs[0].savefig(stem, dpi=150, bbox_inches="tight")
            print(f"Saved: {stem}")
        else:
            for i, fig in enumerate(figs, 1):
                out = stem.with_stem(f"{stem.stem}_{i}")
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"Saved: {out}")
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
