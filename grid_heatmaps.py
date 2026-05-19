"""
5×5 Grid Heatmaps of CH2:CH3 Ratios per Dish
==============================================
Reads ch2_ch3_ratios.csv produced by ch2_ch3_ratio_analysis.py and
renders one spatial heatmap per dish (rows 1-5 × columns A-E),
grouped into one figure per sample group (Opaque / Clear / Yellow).
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

INPUT_CSV  = "ch2_ch3_ratios.csv"
OUTPUT_DIR = "."   # figures saved here

ROWS = list("12345")
COLS = list("ABCDE")

# Shared colour scale limits across all dishes for direct comparison
VMIN, VMAX = 0.6, 2.0
PP_REF = 0.8


def build_grid(subdf):
    """Return 5×5 array (row=1-5, col=A-E) of ratios, NaN where missing."""
    grid = np.full((5, 5), np.nan)
    for _, row in subdf.iterrows():
        gp = str(row["grid_pos"]).upper().strip()
        if len(gp) < 2:
            continue
        try:
            r = ROWS.index(gp[0])
            c = COLS.index(gp[1])
            grid[r, c] = row["ch2_ch3_ratio"]
        except ValueError:
            continue
    return grid


def plot_group(group_df, group_name, out_path):
    dishes = sorted(group_df["dish_id"].unique(),
                    key=lambda d: (len(d), d))  # natural sort

    n = len(dishes)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.2, nrows * 2.4),
        squeeze=False
    )

    cmap = plt.cm.RdYlBu_r   # blue=low (PP-like) → red=high (PE-like)
    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

    for idx, dish in enumerate(dishes):
        ax = axes[idx // ncols][idx % ncols]
        subdf = group_df[group_df["dish_id"] == dish]
        grid  = build_grid(subdf)
        mean  = np.nanmean(grid)

        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal")

        # Annotate each cell with its ratio value
        for r in range(5):
            for c in range(5):
                val = grid[r, c]
                if not np.isnan(val):
                    text_col = "white" if (val > 1.5 or val < 0.85) else "black"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=5.5, color=text_col, fontweight="bold")

        ax.set_xticks(range(5)); ax.set_xticklabels(COLS, fontsize=7)
        ax.set_yticks(range(5)); ax.set_yticklabels(ROWS, fontsize=7)
        ax.set_title(f"{dish}\nμ={mean:.2f}", fontsize=7.5, pad=3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared colourbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes, orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label("CH₂:CH₃ ratio", fontsize=9)
    cbar.ax.axhline(PP_REF, color="black", linewidth=1.2, linestyle="--")
    cbar.ax.text(2.6, PP_REF, f" PP ref\n ({PP_REF})",
                 va="center", fontsize=7, transform=cbar.ax.transData)

    fig.suptitle(
        f"{group_name} dishes — CH₂:CH₃ spatial heatmaps\n"
        f"(blue = PP-like ≤{PP_REF}, red = PE-contaminated ≥{VMAX})",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    df = pd.read_csv(INPUT_CSV)

    # Only process records with a valid grid position (skip QC etc.)
    df = df[df["grid_pos"].astype(str).str.match(r"^\d[A-E]$", na=False)]

    for group in ["Opaque", "Clear", "Yellow"]:
        gdf = df[df["group"] == group]
        if gdf.empty:
            print(f"No data for group: {group}")
            continue
        out = f"{OUTPUT_DIR}/heatmap_{group.lower()}.png"
        plot_group(gdf, group, out)


if __name__ == "__main__":
    main()
