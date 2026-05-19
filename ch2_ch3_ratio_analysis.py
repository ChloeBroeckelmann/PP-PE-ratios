"""
1376/1462 cm-1 Band Ratio Analysis for PP/PE Contamination Screening
=====================================================================
Broeckelmann, C. - Griffith University / Solving Plastic Waste CRC

METHOD
------
Calculates the 1376/1462 cm-1 peak area ratio from ATR-FTIR spectra
exported from PerkinElmer Spectrum Two as two-column CSVs (cm-1, Absorbance).

  1376 cm-1 : CH3 symmetric bending (umbrella mode) — strongly associated
              with the methyl side-chains of isotactic polypropylene (PP).
              This band is weak or absent in polyethylene (PE).

  1462 cm-1 : CH2 scissoring bending — present in both PP and PE.
              In PE this is the dominant band; in PP it overlaps with a
              CH3 asymmetric bending contribution.

A HIGHER ratio (1376/1462 > ~0.9) indicates PP-like material.
A LOWER  ratio indicates increasing PE content (PE has little 1376 signal).

This approach is documented in:
  ASTM D7399-18. Standard Practice for Determination of Certain Polyolefin
  Plastic Materials Using Infrared Spectrophotometry. ASTM International, 2018.

  Gall, M. et al. (2020). Determination of the recycling potential of plastic
  packaging based on near infrared spectroscopy. Waste Management, 110, 43-52.

  Serranti, S. et al. (2011). Characterization of post-consumer polyolefin
  wastes by hyperspectral imaging for quality control in recycling processes.
  Waste Management, 31(11), 2217-2227.

QUALITATIVE PE FLAG
-------------------
Spectra are also checked for a peak near 720 cm-1 (CH2 rocking, n >= 4
consecutive methylenes). This band is characteristic of long-chain PE and
is absent or very weak in PP. A positive flag is reported as a binary
column (720cm_PE_flag) alongside the ratio.

Baseline correction: local two-point linear interpolation between the
absorbance values at the outer edges of each integration window.

Usage:
  1. Set INPUT_FOLDER to the directory containing your CSV files.
  2. Run: python ch2_ch3_ratio_analysis.py

Outputs:
  - fingerprint_ratios.csv          : per-spectrum results
  - fingerprint_ratios_dish_summary.csv : per-dish mean and SD
  - fingerprint_ratios_plot.png     : grouped scatter plot by dish
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
INPUT_FOLDER = "/Users/chloebroeckelmann/Desktop/Claude/FTIR CSV files"
OUTPUT_CSV   = "fingerprint_ratios.csv"
PLOT_OUTPUT  = "fingerprint_ratios_plot.png"

# Integration windows (cm-1) — valley-to-valley boundaries confirmed by
# inspection of actual spectra. Centred on ASTM D7399-18 band positions.
BAND_1376_WINDOW = (1364, 1397)   # CH3 symmetric bend (PP marker)
BAND_1462_WINDOW = (1428, 1483)   # CH2 scissoring (PP + PE)

# Qualitative PE flag: CH2 rocking band present in long-chain PE
BAND_720_CENTRE  = 720
BAND_720_TOL     = 10             # ± cm-1

# ─────────────────────────────────────────────────────────────────────────────


def load_spectrum(filepath):
    """Load PerkinElmer CSV, skip one header row, return (wavenumber, absorbance)."""
    df = pd.read_csv(filepath, skiprows=1, names=["wavenumber", "absorbance"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("wavenumber")
    return df["wavenumber"].values, df["absorbance"].values


def local_baseline_area(wn, ab, wn_low, wn_high):
    """
    Integrate baseline-corrected peak area between wn_low and wn_high.
    Baseline: straight line between absorbance at window edges (two-point).
    Integration: trapezoid rule. Negative area clipped to zero.
    """
    mask   = (wn >= wn_low) & (wn <= wn_high)
    wn_seg = wn[mask]
    ab_seg = ab[mask]

    if len(wn_seg) < 3:
        return np.nan

    ab_baseline = np.interp(wn_seg,
                            [wn_seg[0], wn_seg[-1]],
                            [ab_seg[0], ab_seg[-1]])
    ab_corrected = np.clip(ab_seg - ab_baseline, 0, None)
    return trapezoid(ab_corrected, wn_seg)


def flag_720(wn, ab):
    """
    Return True if a discernible peak exists at 720 ± 10 cm-1.
    Criterion: maximum absorbance in window exceeds the window-edge mean
    by more than 10 % of the local range (simple prominence check).
    """
    lo, hi = BAND_720_CENTRE - BAND_720_TOL, BAND_720_CENTRE + BAND_720_TOL
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 3:
        return False
    ab_seg   = ab[mask]
    baseline = np.mean([ab_seg[0], ab_seg[-1]])
    prominence = ab_seg.max() - baseline
    local_range = ab_seg.max() - ab_seg.min()
    return bool(prominence > 0.1 * local_range and prominence > 1e-4)


def parse_sample_info(filename):
    """
    Extract dish ID, group, and grid position from filename.
    Handles: KDC1(OPAQUE)_1A.csv  KDY12_3B.csv  KDC5_2C.csv
    Group inferred from explicit label (OPAQUE/YELLOW/CLEAR) or prefix
    (KDC → Clear, KDY → Yellow).
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    group = "Unknown"
    for label in ["OPAQUE", "YELLOW", "CLEAR"]:
        if label in base.upper():
            group = label.capitalize()
            break
    if group == "Unknown":
        prefix = re.match(r"^KD([A-Z])", base.upper())
        if prefix:
            group = {"C": "Clear", "Y": "Yellow"}.get(prefix.group(1), "Unknown")

    dish_id_match = re.match(r"^([A-Za-z0-9]+)", base)
    dish_id = dish_id_match.group(1) if dish_id_match else base

    grid_pos = ""
    last_part = base.split("_")[-1]
    if re.match(r"^\d+[A-E]$", last_part, re.IGNORECASE):
        grid_pos = last_part.upper()

    return dish_id, group, grid_pos


def main():
    pattern = os.path.join(INPUT_FOLDER, "*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No CSV files found in '{INPUT_FOLDER}'. Check INPUT_FOLDER.")
        return

    records = []

    for fpath in files:
        try:
            wn, ab = load_spectrum(fpath)
        except Exception as e:
            print(f"  Skipped {os.path.basename(fpath)}: {e}")
            continue

        area_1376 = local_baseline_area(wn, ab, *BAND_1376_WINDOW)
        area_1462 = local_baseline_area(wn, ab, *BAND_1462_WINDOW)

        if area_1462 and area_1462 > 0 and not np.isnan(area_1376):
            ratio = area_1376 / area_1462
        else:
            ratio = np.nan

        pe_flag = flag_720(wn, ab)
        dish_id, group, grid_pos = parse_sample_info(fpath)

        records.append({
            "file":          os.path.basename(fpath),
            "dish_id":       dish_id,
            "group":         group,
            "grid_pos":      grid_pos,
            "area_1376":     round(area_1376, 6) if not np.isnan(area_1376) else np.nan,
            "area_1462":     round(area_1462, 6) if not np.isnan(area_1462) else np.nan,
            "ratio_1376_1462": round(ratio, 4)   if not np.isnan(ratio)     else np.nan,
            "720cm_PE_flag": pe_flag,
        })

    if not records:
        print("No spectra could be processed.")
        return

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-spectrum results saved to: {OUTPUT_CSV}")
    print(df[["dish_id", "group", "grid_pos",
              "ratio_1376_1462", "720cm_PE_flag"]].to_string(index=False))

    # ── Per-dish summary ──────────────────────────────────────────────────────
    summary = (
        df.groupby(["dish_id", "group"])
        .agg(
            mean_ratio =("ratio_1376_1462", "mean"),
            sd_ratio   =("ratio_1376_1462", "std"),
            n          =("ratio_1376_1462", "count"),
            pe_flag_n  =("720cm_PE_flag",   "sum"),
        )
        .reset_index()
        .round({"mean_ratio": 4, "sd_ratio": 4})
    )
    summary_path = OUTPUT_CSV.replace(".csv", "_dish_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nPer-dish summary saved to: {summary_path}")
    print(summary.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    group_colors = {
        "Clear":   "#378ADD",
        "Yellow":  "#EF9F27",
        "Opaque":  "#888780",
        "Unknown": "#D4537E",
    }
    group_order = ["Clear", "Yellow", "Opaque", "Unknown"]

    # Sort summary so x-axis runs Clear → Yellow → Opaque
    summary["_sort"] = summary["group"].map(
        {g: i for i, g in enumerate(group_order)}
    ).fillna(99)
    summary = summary.sort_values(["_sort", "dish_id"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    x_pos   = 0
    x_ticks = []
    x_labels= []

    for _, row in summary.iterrows():
        color   = group_colors.get(row["group"], "#888")
        # Per-spectrum points for this dish
        dish_pts = df[df["dish_id"] == row["dish_id"]]["ratio_1376_1462"].dropna()
        jitter   = np.random.default_rng(42).uniform(-0.18, 0.18, len(dish_pts))
        ax.scatter(
            x_pos + jitter, dish_pts,
            color=color, alpha=0.45, s=18, zorder=3
        )
        # Mean ± SD
        ax.errorbar(
            x_pos, row["mean_ratio"],
            yerr=row["sd_ratio"],
            fmt="o", color=color, markeredgecolor="white",
            markeredgewidth=0.8, markersize=7,
            elinewidth=1.2, capsize=3, zorder=4
        )
        x_ticks.append(x_pos)
        x_labels.append(row["dish_id"])
        x_pos += 1

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
    ax.set_ylabel("1376 / 1462 cm⁻¹ band area ratio", fontsize=11)
    ax.set_title(
        "Fingerprint region PP/PE ratio by dish  "
        "(1376 cm⁻¹ CH₃ bend / 1462 cm⁻¹ CH₂ bend)\n"
        "Higher ratio = more PP-like  ·  Lower ratio = more PE-like  "
        "·  ref: ASTM D7399-18",
        fontsize=10
    )

    # Group colour legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=group_colors[g], label=g)
        for g in group_order if g in df["group"].values
    ]
    ax.legend(handles=handles, fontsize=10, frameon=True)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"\nPlot saved to: {PLOT_OUTPUT}")
    plt.close()

    # ── 720 cm-1 flag summary ─────────────────────────────────────────────────
    flag_summary = (
        df.groupby("group")["720cm_PE_flag"]
        .agg(flagged="sum", total="count")
        .assign(pct_flagged=lambda x: (x["flagged"] / x["total"] * 100).round(1))
    )
    print("\n720 cm-1 PE flag summary (qualitative):")
    print(flag_summary.to_string())


if __name__ == "__main__":
    main()
