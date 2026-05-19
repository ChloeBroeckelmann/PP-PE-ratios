"""
Fingerprint Ratio Explorer — 1376/1462 cm-1 PP/PE Screening
============================================================
Run with: streamlit run app.py
Method: ASTM D7399-18 fingerprint region band ratio
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import streamlit as st
from scipy.integrate import trapezoid
from scipy import stats as scipy_stats

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "fingerprint_ratios.csv"
SUMMARY_CSV  = "fingerprint_ratios_dish_summary.csv"
CALIB_PE_CSV = "calibration_PE.csv"
CALIB_UH_CSV = "calibration_UHMWPE.csv"
CALIB_MODEL  = "calibration_models.txt"
FTIR_FOLDER  = "FTIR CSV files"

BAND_1376 = (1364, 1397)
BAND_1462 = (1428, 1483)
BAND_720_CENTRE = 720
BAND_720_TOL    = 10
RATIO_COL   = "ratio_1376_1462"
VMIN, VMAX  = 0.3, 0.9

ROWS  = list("12345")
COLS  = list("ABCDE")
GROUP_ORDER  = ["Clear", "Yellow", "Opaque"]
GROUP_COLORS = {"Clear": "#378ADD", "Yellow": "#EF9F27", "Opaque": "#888780"}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_CSV)

@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_CSV)

@st.cache_data
def load_calib():
    pe = pd.read_csv(CALIB_PE_CSV) if os.path.exists(CALIB_PE_CSV) else None
    uh = pd.read_csv(CALIB_UH_CSV) if os.path.exists(CALIB_UH_CSV) else None
    return pe, uh

@st.cache_data
def load_calib_models():
    if not os.path.exists(CALIB_MODEL):
        return {}
    models, current = {}, None
    with open(CALIB_MODEL) as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                current = line.strip("[]")
                models[current] = {}
            elif "=" in line and current:
                k, v = line.split("=", 1)
                models[current][k.strip()] = float(v.strip())
    return models

@st.cache_data
def load_spectrum(filepath):
    df = pd.read_csv(filepath, skiprows=1, names=["wavenumber", "absorbance"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_values("wavenumber")
    return df["wavenumber"].values, df["absorbance"].values

def local_baseline(wn, ab, lo, hi):
    mask = (wn >= lo) & (wn <= hi)
    wn_s, ab_s = wn[mask], ab[mask]
    if len(wn_s) < 3:
        return wn_s, ab_s, np.zeros_like(ab_s)
    bl = np.interp(wn_s, [wn_s[0], wn_s[-1]], [ab_s[0], ab_s[-1]])
    return wn_s, ab_s, bl

def get_area(wn, ab, lo, hi):
    wn_s, ab_s, bl = local_baseline(wn, ab, lo, hi)
    if len(wn_s) < 3:
        return 0.0
    return float(trapezoid(np.clip(ab_s - bl, 0, None), wn_s))

def build_grid(subdf):
    grid = np.full((5, 5), np.nan)
    for _, row in subdf.iterrows():
        gp = str(row["grid_pos"]).upper().strip()
        if len(gp) < 2:
            continue
        try:
            r, c = ROWS.index(gp[0]), COLS.index(gp[1])
            grid[r, c] = row[RATIO_COL]
        except ValueError:
            continue
    return grid

def find_csv(dish_id, grid_pos):
    matches = glob.glob(os.path.join(FTIR_FOLDER, f"*{dish_id}*_{grid_pos}.csv"))
    return matches[0] if matches else None

def predict_poly(coeffs, x):
    return np.polyval(coeffs, x)

def invert_poly(coeffs, y_val):
    a, b, c = coeffs
    roots = np.roots([a, b, c - y_val])
    real = [r.real for r in roots if abs(r.imag) < 1e-6 and 0 <= r.real <= 100]
    return round(real[0], 1) if real else np.nan

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PP/PE Ratio Explorer", layout="wide")
st.title("PP/PE Fingerprint Ratio Explorer")
st.caption("1376 / 1462 cm⁻¹ band area ratio · ASTM D7399-18")

df      = load_results()
summary = load_summary()
calib_pe, calib_uh = load_calib()
models  = load_calib_models()
groups_avail = [g for g in GROUP_ORDER if g in df["group"].unique()]

tabs = st.tabs([
    "📊  Compare",
    "🗺️  Spatial Heatmap",
    "📈  Calibration",
    "🔬  Spectrum Viewer",
    "💡  How it works",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Compare (All samples / By group / Within dish)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Compare ratios across the dataset")
    st.caption("Higher ratio = more PP-like · Lower ratio = more PE contamination")

    level = st.radio(
        "Compare at level:",
        ["All samples", "By group", "Within dish"],
        horizontal=True
    )
    st.divider()

    plot_df = df[df["group"].isin(GROUP_ORDER)].copy()

    # ── All samples ───────────────────────────────────────────────────────────
    if level == "All samples":
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**All spectra — distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            data = plot_df[RATIO_COL].dropna().values
            vp = ax.violinplot(data, positions=[0], showmedians=True,
                               showextrema=True, widths=0.5)
            for pc in vp["bodies"]:
                pc.set_facecolor("#888"); pc.set_alpha(0.5)
            for key in ("cmedians","cbars","cmaxes","cmins"):
                vp[key].set_color("#444")
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(data))
            ax.scatter(jitter, data, color="#555", alpha=0.2, s=6, zorder=3)
            ax.set_xticks([0]); ax.set_xticklabels(["All samples"])
            ax.set_ylabel("1376 / 1462 ratio")
            ax.grid(axis="y", alpha=0.2)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            st.markdown("**Summary statistics**")
            stats = plot_df[RATIO_COL].describe().round(4)
            st.dataframe(stats.to_frame("value"), width="stretch")
            st.metric("Total spectra", len(plot_df))
            st.metric("Total dishes", plot_df["dish_id"].nunique())

            if "720cm_PE_flag" in plot_df.columns:
                n_flag = plot_df["720cm_PE_flag"].sum()
                st.metric("720 cm⁻¹ PE flag positive",
                          f"{n_flag} / {len(plot_df)} spectra")

    # ── By group ──────────────────────────────────────────────────────────────
    elif level == "By group":
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("**Violin plot by group**")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for i, grp in enumerate(GROUP_ORDER):
                gdata = plot_df[plot_df["group"] == grp][RATIO_COL].dropna().values
                if len(gdata) == 0:
                    continue
                vp = ax.violinplot(gdata, positions=[i], showmedians=True,
                                   showextrema=True, widths=0.6)
                color = GROUP_COLORS[grp]
                for pc in vp["bodies"]:
                    pc.set_facecolor(color); pc.set_alpha(0.55)
                for key in ("cmedians","cbars","cmaxes","cmins"):
                    vp[key].set_color(color)
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(gdata))
                ax.scatter(np.full(len(gdata), i) + jitter, gdata,
                           color=color, alpha=0.3, s=8, zorder=3)
            ax.set_xticks(range(len(GROUP_ORDER)))
            ax.set_xticklabels(GROUP_ORDER)
            ax.set_ylabel("1376 / 1462 ratio")
            ax.grid(axis="y", alpha=0.2)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            st.markdown("**Mean ± SD per dish, coloured by group**")
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            sum_plot = summary[summary["group"].isin(GROUP_ORDER)].copy()
            sum_plot["_s"] = sum_plot["group"].map(
                {g: i for i, g in enumerate(GROUP_ORDER)})
            sum_plot = sum_plot.sort_values(["_s", "dish_id"])
            ax2.bar(sum_plot["dish_id"], sum_plot["mean_ratio"],
                    yerr=sum_plot["sd_ratio"],
                    color=[GROUP_COLORS[g] for g in sum_plot["group"]],
                    alpha=0.85,
                    error_kw={"elinewidth": 0.7, "capsize": 1.5})
            handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
                       for g in GROUP_ORDER]
            ax2.legend(handles=handles, fontsize=8)
            ax2.set_ylabel("Mean 1376 / 1462 ratio")
            ax2.tick_params(axis="x", rotation=90, labelsize=5.5)
            ax2.grid(axis="y", alpha=0.2)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("**Group summary table**")
        grp_summary = (
            plot_df.groupby("group")[RATIO_COL]
            .agg(n="count", mean="mean", sd="std",
                 median="median", min="min", max="max")
            .round(4)
            .loc[[g for g in GROUP_ORDER if g in plot_df["group"].unique()]]
        )
        st.dataframe(grp_summary, width="stretch")

        # Statistical test
        st.markdown("**One-way ANOVA across groups**")
        group_data = [
            plot_df[plot_df["group"] == g][RATIO_COL].dropna().values
            for g in GROUP_ORDER if g in plot_df["group"].unique()
        ]
        if len(group_data) >= 2:
            f_stat, p_val = scipy_stats.f_oneway(*group_data)
            sig = "✅ Significant (p < 0.05)" if p_val < 0.05 else "❌ Not significant"
            col1, col2, col3 = st.columns(3)
            col1.metric("F-statistic", f"{f_stat:.3f}")
            col2.metric("p-value", f"{p_val:.4f}")
            col3.metric("Result", sig)

    # ── Within dish ───────────────────────────────────────────────────────────
    elif level == "Within dish":
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            sel_group = st.selectbox("Group", groups_avail, key="wd_group")
        with d_col2:
            sel_dishes = sorted(
                df[df["group"] == sel_group]["dish_id"].unique(),
                key=lambda d: (len(d), d)
            )
            sel_dish = st.selectbox("Dish", sel_dishes, key="wd_dish")

        dish_df = plot_df[plot_df["dish_id"] == sel_dish].copy()
        dish_df["pos"] = dish_df["grid_pos"].astype(str)

        dish_sum = summary[summary["dish_id"] == sel_dish]
        if not dish_sum.empty:
            mean_r = dish_sum["mean_ratio"].values[0]
            sd_r   = dish_sum["sd_ratio"].values[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Dish", sel_dish)
            c2.metric("Mean ratio", f"{mean_r:.4f}", delta=f"±{sd_r:.4f} SD")
            c3.metric(
                "Status",
                "✅ PP-like" if mean_r >= 0.50 else "⚠️ PE contamination likely"
            )

        st.divider()
        gc1, gc2 = st.columns(2, gap="large")

        with gc1:
            st.markdown(f"**All 25 grid positions — {sel_dish}**")
            fig, ax = plt.subplots(figsize=(7, 4))
            sorted_df = dish_df.sort_values("pos")
            colors_pt = [
                "#E05C5C" if r < 0.55 else "#EF9F27" if r < 0.65 else "#378ADD"
                for r in sorted_df[RATIO_COL]
            ]
            ax.bar(sorted_df["pos"], sorted_df[RATIO_COL],
                   color=colors_pt, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.axhline(mean_r, color="#333", lw=1.2, ls="--",
                       label=f"Dish mean ({mean_r:.4f})")
            ax.set_xlabel("Grid position"); ax.set_ylabel("1376 / 1462 ratio")
            ax.set_title(f"{sel_dish} — ratio at each grid position")
            ax.tick_params(axis="x", rotation=90, labelsize=7)
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with gc2:
            st.markdown(f"**Spatial heatmap — {sel_dish}**")
            grid  = build_grid(dish_df)
            cmap  = plt.cm.RdYlBu
            norm  = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
            fig2, ax2 = plt.subplots(figsize=(4.5, 4))
            ax2.imshow(grid, cmap=cmap, norm=norm, aspect="equal")
            for r in range(5):
                for c in range(5):
                    val = grid[r, c]
                    if not np.isnan(val):
                        tc = "white" if abs(val - 0.6) > 0.15 else "black"
                        ax2.text(c, r, f"{val:.3f}", ha="center", va="center",
                                 fontsize=7.5, color=tc, fontweight="bold")
            ax2.set_xticks(range(5)); ax2.set_xticklabels(COLS)
            ax2.set_yticks(range(5)); ax2.set_yticklabels(ROWS)
            cbar = fig2.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label("1376/1462 ratio")
            plt.tight_layout(); st.pyplot(fig2); plt.close()
            st.caption("Blue = PP-like · Red = PE-like")

        st.markdown("**Grid position data**")
        st.dataframe(
            dish_df[["grid_pos", "area_1376", "area_1462",
                     RATIO_COL, "720cm_PE_flag"]]
            .sort_values("grid_pos")
            .round(5)
            .reset_index(drop=True),
            width="stretch"
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Per-spectrum CSV",
                           df.to_csv(index=False).encode(),
                           "fingerprint_ratios_full.csv", "text/csv")
    with c2:
        st.download_button("⬇️ Dish summary CSV",
                           summary.to_csv(index=False).encode(),
                           "fingerprint_ratios_dish_summary.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Spatial Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.sidebar.header("Select sample")
    group = st.sidebar.selectbox("Group", groups_avail)
    group_dishes = sorted(df[df["group"] == group]["dish_id"].unique(),
                          key=lambda d: (len(d), d))
    dish    = st.sidebar.selectbox("Dish", group_dishes)
    dish_df = df[(df["group"] == group) & (df["dish_id"] == dish)]
    grid    = build_grid(dish_df)

    dish_sum = summary[summary["dish_id"] == dish]
    if not dish_sum.empty:
        mean_r = dish_sum["mean_ratio"].values[0]
        sd_r   = dish_sum["sd_ratio"].values[0]
        flag_n = dish_sum["pe_flag_n"].values[0] if "pe_flag_n" in dish_sum else "—"
        status = "✅ PP-like" if mean_r >= 0.50 else "⚠️ PE contamination likely"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dish", dish); c2.metric("Group", group)
        c3.metric("Mean 1376/1462", f"{mean_r:.4f}", delta=f"±{sd_r:.4f} SD")
        c4.metric("Status", status)

    st.divider()
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.subheader(f"{dish} — spatial grid")
        cmap = plt.cm.RdYlBu
        norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
        fig, ax = plt.subplots(figsize=(4, 3.8))
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal")
        for r in range(5):
            for c in range(5):
                val = grid[r, c]
                if not np.isnan(val):
                    tc = "white" if abs(val - 0.6) > 0.15 else "black"
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=7.5, color=tc, fontweight="bold")
        ax.set_xticks(range(5)); ax.set_xticklabels(COLS)
        ax.set_yticks(range(5)); ax.set_yticklabels(ROWS)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("1376/1462 ratio")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Blue = PP-like (high ratio) · Red = PE-like (low ratio)")

        st.markdown("**Inspect a grid position**")
        row_pick = st.selectbox("Row", ROWS, key="row")
        col_pick = st.selectbox("Column", COLS, key="col")
        selected_pos = f"{row_pick}{col_pick}"

    with right:
        st.subheader(f"Spectrum — {dish} {selected_pos}")
        csv_path = find_csv(dish, selected_pos)
        if csv_path:
            wn, ab = load_spectrum(csv_path)
            zoom_mask = (wn >= 1300) & (wn <= 1550)
            wn_z, ab_z = wn[zoom_mask], ab[zoom_mask]
            wn_a, ab_a, bl_a = local_baseline(wn, ab, *BAND_1376)
            wn_b, ab_b, bl_b = local_baseline(wn, ab, *BAND_1462)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(wn_z, ab_z, color="#333", lw=1.2, label="Spectrum")
            ax2.fill_between(wn_a, bl_a, np.maximum(ab_a, bl_a),
                             alpha=0.4, color="#7B5EA7",
                             label=f"1376 cm⁻¹ {BAND_1376}")
            ax2.plot(wn_a, bl_a, color="#7B5EA7", lw=0.8, ls="--")
            ax2.fill_between(wn_b, bl_b, np.maximum(ab_b, bl_b),
                             alpha=0.4, color="#E05C5C",
                             label=f"1462 cm⁻¹ {BAND_1462}")
            ax2.plot(wn_b, bl_b, color="#E05C5C", lw=0.8, ls="--")
            row_data = dish_df[dish_df["grid_pos"] == selected_pos]
            if not row_data.empty:
                ratio = row_data[RATIO_COL].values[0]
                a1376 = row_data["area_1376"].values[0]
                a1462 = row_data["area_1462"].values[0]
                flag  = row_data["720cm_PE_flag"].values[0] \
                        if "720cm_PE_flag" in row_data else "—"
                ax2.set_title(
                    f"1376={a1376:.4f}  |  1462={a1462:.4f}  "
                    f"|  Ratio={ratio:.4f}  |  720 flag={'Yes' if flag else 'No'}",
                    fontsize=8.5)
            ax2.set_xlabel("Wavenumber (cm⁻¹)"); ax2.set_ylabel("Absorbance")
            ax2.invert_xaxis(); ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
            plt.tight_layout(); st.pyplot(fig2); plt.close()
            st.caption(f"`{os.path.basename(csv_path)}`")
        else:
            st.info(f"No CSV found for {dish} {selected_pos}.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Calibration
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Calibration curves — 1376 / 1462 cm⁻¹ method")
    st.markdown(
        "Two separate quadratic calibration curves fitted to polymer mixture "
        "standards. Boundaries adjusted to valley-to-valley positions confirmed "
        "by spectral inspection (centred on ASTM D7399-18 band positions)."
    )

    if calib_pe is None or calib_uh is None:
        st.warning("Run `calibration.py` to generate calibration data.")
    else:
        x_fit = np.linspace(0, 90, 300)
        sample_ratios = df[RATIO_COL].dropna()

        col1, col2 = st.columns(2, gap="large")

        for col, calib, label, color, key in [
            (col1, calib_pe, "PP + PE (regular)",  "#E05C5C", "PE"),
            (col2, calib_uh, "PP + UHMWPE",        "#7B5EA7", "UHMWPE"),
        ]:
            with col:
                st.markdown(f"**{label}**")
                x = calib["pe_pct"].values
                y = calib["ch2_ch3_ratio"].values
                coeffs = np.polyfit(x, y, deg=2)
                y_pred = np.polyval(coeffs, x)
                r2 = 1 - np.sum((y-y_pred)**2) / np.sum((y-np.mean(y))**2)
                a, b, c = coeffs

                y_fit = np.polyval(coeffs, x_fit)

                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.scatter(x, y, color="#333", s=30, zorder=4, label="Standards")
                ax.plot(x_fit, y_fit, color=color, lw=2,
                        label=f"{a:.5f}x² + {b:.4f}x + {c:.3f}\nR² = {r2:.3f}")
                ax.axhspan(sample_ratios.min(), sample_ratios.max(),
                           alpha=0.1, color="#378ADD",
                           label=f"Your samples "
                                 f"({sample_ratios.min():.3f}–"
                                 f"{sample_ratios.max():.3f})")
                ax.set_xlabel("PE content (% by weight)")
                ax.set_ylabel("1376 / 1462 ratio")
                ax.legend(fontsize=8); ax.grid(alpha=0.2)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.metric("R²", f"{r2:.3f}")
                st.download_button(
                    f"⬇️ {key} calibration data",
                    calib.to_csv(index=False).encode(),
                    f"calibration_{key}.csv", "text/csv",
                    key=f"dl_{key}"
                )

        st.divider()
        st.subheader("Estimated % PE in your samples")
        st.caption(
            "Estimated by inverting the calibration equations. "
            "Treat as indicative — calibration standards were measured on a "
            "different instrument to the samples."
        )
        if os.path.exists("ch2_ch3_ratios_with_PE_estimate.csv"):
            est_df = pd.read_csv("ch2_ch3_ratios_with_PE_estimate.csv")
            if "pe_pct_PE_curve" in est_df.columns:
                est = (
                    est_df[est_df["group"].isin(GROUP_ORDER)]
                    .groupby("group")[["pe_pct_PE_curve","pe_pct_UHMWPE_curve"]]
                    .agg(["mean","std"]).round(1)
                )
                est.columns = ["PE mean (%)","PE SD","UHMWPE mean (%)","UHMWPE SD"]
                st.dataframe(est, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Spectrum Viewer (multi-select overlay)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Spectrum viewer — overlay multiple spectra")
    st.markdown(
        "Select any combination of groups, dishes, and grid positions to "
        "overlay their spectra on the same plot."
    )

    all_pos = [f"{r}{c}" for r in ROWS for c in COLS]

    # ── Quick-select callbacks ────────────────────────────────────────────────
    def _all_groups():
        st.session_state["sv_groups"] = list(groups_avail)

    def _groups_changed():
        valid = set(df[df["group"].isin(
            st.session_state.get("sv_groups", []))]["dish_id"])
        st.session_state["sv_dishes"] = [
            d for d in st.session_state.get("sv_dishes", []) if d in valid
        ]

    def _all_dishes():
        cur = st.session_state.get("sv_groups", [])
        st.session_state["sv_dishes"] = sorted(
            df[df["group"].isin(cur)]["dish_id"].unique(),
            key=lambda d: (len(d), d)
        )

    def _group_dishes(g):
        st.session_state["sv_groups"] = [g]
        st.session_state["sv_dishes"] = sorted(
            df[df["group"] == g]["dish_id"].unique(),
            key=lambda d: (len(d), d)
        )

    def _all_pos_cb():
        st.session_state["sv_positions"] = list(all_pos)

    def _row_cb(r):
        st.session_state["sv_positions"] = [f"{r}{c}" for c in COLS]

    def _col_cb(c):
        st.session_state["sv_positions"] = [f"{r}{c}" for r in ROWS]

    sel1, sel2, sel3 = st.columns(3)

    with sel1:
        st.markdown("**Groups**")
        st.button("Select all groups", key="sv_btn_all_grp",
                  on_click=_all_groups, width="stretch")
        sv_groups = st.multiselect(
            "Group(s)", groups_avail,
            default=[groups_avail[0]],
            key="sv_groups",
            on_change=_groups_changed,
        )

    with sel2:
        st.markdown("**Dishes**")
        grp_btn_cols = st.columns(len(groups_avail))
        for i, g in enumerate(groups_avail):
            grp_btn_cols[i].button(
                f"All {g}", key=f"sv_btn_{g}",
                on_click=_group_dishes, args=(g,),
                width="stretch",
            )
        st.button("All shown dishes", key="sv_btn_all_dishes",
                  on_click=_all_dishes, width="stretch")
        available_dishes = sorted(
            df[df["group"].isin(sv_groups)]["dish_id"].unique(),
            key=lambda d: (len(d), d)
        ) if sv_groups else []
        sv_dishes = st.multiselect(
            "Dish(es)", available_dishes,
            default=available_dishes[:1] if available_dishes else [],
            key="sv_dishes",
        )

    with sel3:
        st.markdown("**Grid positions**")
        st.button("All 25 positions", key="sv_btn_all_pos",
                  on_click=_all_pos_cb, width="stretch")
        row_btns = st.columns(5)
        for i, r in enumerate(ROWS):
            row_btns[i].button(f"Row {r}", key=f"sv_row_{r}",
                               on_click=_row_cb, args=(r,),
                               width="stretch")
        col_btns = st.columns(5)
        for i, c in enumerate(COLS):
            col_btns[i].button(f"Col {c}", key=f"sv_col_{c}",
                               on_click=_col_cb, args=(c,),
                               width="stretch")
        sv_positions = st.multiselect(
            "Grid position(s)", all_pos,
            default=["1A", "3C", "5E"],
            key="sv_positions",
        )

    # ── Peak ratio selector ───────────────────────────────────────────────────
    st.divider()
    RATIO_PRESETS = {
        "1376 / 1462 cm⁻¹ — CH₃/CH₂ bend (ASTM D7399-18)": {
            "band_a": (1364, 1397), "band_b": (1428, 1483),
            "label_a": "1376 cm⁻¹", "label_b": "1462 cm⁻¹",
            "view": (650, 1550),   # wide enough to include 720
        },
        "998 / 973 cm⁻¹ — PP crystallinity index": {
            "band_a": (985, 1010), "band_b": (960, 985),
            "label_a": "998 cm⁻¹", "label_b": "973 cm⁻¹",
            "view": (650, 1100),   # 720 visible on left
        },
        "2920 / 2950 cm⁻¹ — CH₂/CH₃ C-H stretching": {
            "band_a": (2905, 2930), "band_b": (2940, 2965),
            "label_a": "2920 cm⁻¹", "label_b": "2950 cm⁻¹",
            "view": (2870, 3000),  # 720 too far away for this one
        },
        "1168 / 1462 cm⁻¹ — PP isotactic CH₃ twist / CH₂ scissor": {
            "band_a": (1150, 1185), "band_b": (1428, 1483),
            "label_a": "1168 cm⁻¹", "label_b": "1462 cm⁻¹",
            "view": (650, 1550),   # wide enough to include 720
        },
        "Custom": None,
    }

    r1, r2 = st.columns([2, 3])
    sel_ratio = r1.selectbox(
        "Peak ratio to calculate",
        list(RATIO_PRESETS.keys()), key="sv_ratio_preset",
    )

    if sel_ratio == "Custom":
        rc1, rc2, rc3, rc4 = st.columns(4)
        lo_a = rc1.number_input("Band A low (cm⁻¹)", value=1364, step=1, key="sv_lo_a")
        hi_a = rc2.number_input("Band A high (cm⁻¹)", value=1397, step=1, key="sv_hi_a")
        lo_b = rc3.number_input("Band B low (cm⁻¹)", value=1428, step=1, key="sv_lo_b")
        hi_b = rc4.number_input("Band B high (cm⁻¹)", value=1483, step=1, key="sv_hi_b")
        band_a = (int(lo_a), int(hi_a))
        band_b = (int(lo_b), int(hi_b))
        ctr_a  = int((lo_a + hi_a) / 2)
        ctr_b  = int((lo_b + hi_b) / 2)
        sv_label_a = f"{ctr_a} cm⁻¹"
        sv_label_b = f"{ctr_b} cm⁻¹"
        sv_ratio_label = f"{ctr_a} / {ctr_b} cm⁻¹"
        pad = max(50, int((max(hi_a, hi_b) - min(lo_a, lo_b)) * 0.15))
        sv_view = (min(lo_a, lo_b) - pad, max(hi_a, hi_b) + pad)
    else:
        preset = RATIO_PRESETS[sel_ratio]
        band_a, band_b = preset["band_a"], preset["band_b"]
        sv_label_a, sv_label_b = preset["label_a"], preset["label_b"]
        sv_ratio_label = sel_ratio.split(" — ")[0]
        sv_view = preset["view"]

    if sv_groups and sv_dishes and sv_positions:
        # Build list of (label, filepath) for selected combos; ratio computed live
        selected = []
        for dish_id in sv_dishes:
            dish_group = df[df["dish_id"] == dish_id]["group"].iloc[0] \
                         if not df[df["dish_id"] == dish_id].empty else "?"
            for pos in sv_positions:
                fpath = find_csv(dish_id, pos)
                selected.append({
                    "label": f"{dish_id} {pos}",
                    "group": dish_group,
                    "path":  fpath,
                })

        found = [s for s in selected if s["path"]]
        missing = [s["label"] for s in selected if not s["path"]]

        if missing:
            st.warning(f"No CSV found for: {', '.join(missing)}")

        if found:
            import matplotlib
            cmap_lines = matplotlib.colormaps["tab10"]
            n = len(found)

            fig, axes = plt.subplots(
                1, 2, figsize=(14, 4.5),
                gridspec_kw={"width_ratios": [2.5, 1]},
            )
            ax_spec, ax_bar = axes

            bar_labels, bar_ratios, bar_colors = [], [], []

            for i, s in enumerate(found):
                wn, ab = load_spectrum(s["path"])
                color = GROUP_COLORS.get(s["group"], cmap_lines(i / max(n - 1, 1)))

                # Normalise to fingerprint-region max so 720 appears at its
                # true relative height (not blown up by local normalisation)
                fp_mask = (wn >= 1300) & (wn <= 1550)
                fp_max  = ab[fp_mask].max() if fp_mask.any() else ab.max()
                norm_factor = fp_max if fp_max > 0 else 1.0

                lo_v, hi_v = sv_view
                zm = (wn >= lo_v) & (wn <= hi_v)
                ab_norm = ab[zm] / norm_factor

                # Compute ratio from raw spectrum
                area_a = get_area(wn, ab, *band_a)
                area_b = get_area(wn, ab, *band_b)
                ratio  = area_a / area_b if area_b > 0 else np.nan

                ax_spec.plot(wn[zm], ab_norm, lw=1.2, color=color, alpha=0.8,
                             label=f"{s['label']} ({ratio:.3f})" if not np.isnan(ratio)
                                   else s["label"])

                bar_labels.append(s["label"])
                bar_ratios.append(ratio)
                bar_colors.append(color)

            # Shade ratio integration windows
            for (lo, hi), col, lbl in [
                (band_a, "#7B5EA7", sv_label_a),
                (band_b, "#E05C5C", sv_label_b),
            ]:
                if lo_v <= lo <= hi_v or lo_v <= hi <= hi_v:
                    ax_spec.axvspan(lo, hi, alpha=0.08, color=col)
                    ax_spec.axvline(lo, color=col, lw=0.5, ls=":")
                    ax_spec.axvline(hi, color=col, lw=0.5, ls=":")
                    ax_spec.text((lo + hi) / 2, 0.02, lbl,
                                 ha="center", color=col, fontsize=7, alpha=0.9)

            # Shade 720 window if it falls inside the current view
            lo720 = BAND_720_CENTRE - BAND_720_TOL
            hi720 = BAND_720_CENTRE + BAND_720_TOL
            if lo_v <= BAND_720_CENTRE <= hi_v:
                ax_spec.axvspan(lo720, hi720, alpha=0.12, color="#2CA05A")
                ax_spec.axvline(BAND_720_CENTRE, color="#2CA05A", lw=1, ls="--")
                ax_spec.text(BAND_720_CENTRE, 0.02, "720 cm⁻¹\n(PE)",
                             ha="center", color="#2CA05A", fontsize=7, alpha=0.9)

            ax_spec.set_xlabel("Wavenumber (cm⁻¹)")
            ax_spec.set_ylabel("Absorbance (normalised to 1300–1550 max)")
            ax_spec.set_title(f"Overlaid spectra — {sv_ratio_label}")
            ax_spec.invert_xaxis()
            ax_spec.legend(fontsize=7, loc="upper right",
                           bbox_to_anchor=(1.0, 1.0))
            ax_spec.grid(alpha=0.15)

            # Bar chart
            valid_ratios = [v if not np.isnan(v) else 0 for v in bar_ratios]
            bars = ax_bar.barh(bar_labels, valid_ratios, color=bar_colors,
                               alpha=0.85, edgecolor="white")
            ax_bar.set_xlabel(sv_ratio_label)
            ax_bar.set_title("Ratio comparison")
            ax_bar.grid(axis="x", alpha=0.2)
            for bar, val in zip(bars, bar_ratios):
                if not np.isnan(val):
                    ax_bar.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                                f"{val:.4f}", va="center", fontsize=8)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Data table
            st.markdown(f"**Selected spectra — {sv_ratio_label} values**")
            table = pd.DataFrame([
                {"Label": s["label"], "Group": s["group"],
                 "Ratio": round(r, 4) if not np.isnan(r) else "—"}
                for s, r in zip(found, bar_ratios)
            ])
            st.dataframe(table, width="stretch")
    else:
        st.info("Select at least one group, dish, and grid position above.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — How it works
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("How the 1376 / 1462 cm⁻¹ ratio works — step by step")

    @st.cache_data
    def load_explainer_spectra():
        xl  = pd.ExcelFile("All polymer mixture spectra FTIR _abs.xlsx")
        df1 = pd.read_excel(xl, sheet_name="All polymer mixture spectra_01",
                            header=None)
        df2 = pd.read_excel(xl, sheet_name="All polymer mixture spectra FTI",
                            header=None)
        wn_pp = pd.to_numeric(df1.iloc[2:, 74], errors="coerce").dropna().values
        ab_pp = pd.to_numeric(df1.iloc[2:, 75], errors="coerce").dropna().values
        n = min(len(wn_pp), len(ab_pp))
        idx = np.argsort(wn_pp[:n])
        wn_pp, ab_pp = wn_pp[:n][idx], ab_pp[:n][idx]

        wn_pe_row = pd.to_numeric(df2.iloc[0, 2:], errors="coerce").values
        ab_pe     = pd.to_numeric(df2.iloc[6, 2:], errors="coerce").values
        n2 = min(len(wn_pe_row), len(ab_pe))
        wn_pe, ab_pe = wn_pe_row[:n2], ab_pe[:n2]
        mask = ~np.isnan(wn_pe) & ~np.isnan(ab_pe)
        wn_pe, ab_pe = wn_pe[mask], ab_pe[mask]
        idx2 = np.argsort(wn_pe)
        return (wn_pp, ab_pp), (wn_pe[idx2], ab_pe[idx2])

    (wn_pp, ab_pp), (wn_pe, ab_pe) = load_explainer_spectra()
    ZOOM = (1300, 1550)
    def norm(ab): return ab / ab.max()

    st.markdown("---")
    st.markdown("### Step 1 — The fingerprint region")
    st.markdown(
        "We look at the **1300–1550 cm⁻¹ fingerprint region**. PP and PE look "
        "noticeably different here."
    )
    fig, ax = plt.subplots(figsize=(9, 3.5))
    zm_pp = (wn_pp >= ZOOM[0]) & (wn_pp <= ZOOM[1])
    zm_pe = (wn_pe >= ZOOM[0]) & (wn_pe <= ZOOM[1])
    ax.plot(wn_pp[zm_pp], norm(ab_pp)[zm_pp], color="#378ADD", lw=1.8,
            label="90% PP standard")
    ax.plot(wn_pe[zm_pe], norm(ab_pe)[zm_pe], color="#E05C5C", lw=1.8,
            label="90% PE standard")
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Absorbance (normalised)")
    ax.invert_xaxis(); ax.legend(fontsize=10); ax.grid(alpha=0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### Step 2 — Two peaks, two stories")
    st.markdown(
        "- 🟣 **1376 cm⁻¹** — CH₃ bending. Strong in PP, nearly absent in PE.\n\n"
        "- 🔴 **1462 cm⁻¹** — CH₂ scissoring. Present in both — acts as a reference."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, wn, ab, color, label in [
        (axes[0], wn_pp, norm(ab_pp), "#378ADD", "90% PP standard"),
        (axes[1], wn_pe, norm(ab_pe), "#E05C5C", "90% PE standard"),
    ]:
        zm = (wn >= ZOOM[0]) & (wn <= ZOOM[1])
        ax.plot(wn[zm], ab[zm], color=color, lw=1.8)
        ax.axvspan(*BAND_1376, alpha=0.18, color="#7B5EA7")
        ax.axvspan(*BAND_1462, alpha=0.18, color="#E05C5C")
        ax.axvline(1376, color="#7B5EA7", lw=1, ls="--")
        ax.axvline(1462, color="#E05C5C", lw=1, ls="--")
        ht = ab[zm].max()
        ax.text(1376, ht*0.55, "1376\nPP marker", ha="center",
                color="#7B5EA7", fontsize=9, fontweight="bold")
        ax.text(1462, ht*0.55, "1462\nreference", ha="center",
                color="#E05C5C", fontsize=9, fontweight="bold")
        ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Absorbance (normalised)")
        ax.set_title(label); ax.invert_xaxis(); ax.grid(alpha=0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### Step 3 — Measure the areas")
    st.markdown(
        "A straight **baseline** (dashed) connects the valley on each side of "
        "the peak. Only the shaded area **above** that line is counted. "
        "The 1376 area almost disappears in PE."
    )
    st.caption(
        "ℹ️ Windows are set valley-to-valley (confirmed by spectral inspection) "
        "so the baseline sits at the foot of each peak."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, wn, ab_raw, color, label in [
        (axes[0], wn_pp, ab_pp, "#378ADD", "90% PP standard"),
        (axes[1], wn_pe, ab_pe, "#E05C5C", "90% PE standard"),
    ]:
        zm = (wn >= ZOOM[0]) & (wn <= ZOOM[1])
        ax.plot(wn[zm], ab_raw[zm], color=color, lw=1.5, zorder=3)
        for (lo, hi), col, name in [
            (BAND_1376, "#7B5EA7", "1376"),
            (BAND_1462, "#E05C5C", "1462"),
        ]:
            wn_s, ab_s, bl = local_baseline(wn, ab_raw, lo, hi)
            area = get_area(wn, ab_raw, lo, hi)
            ax.fill_between(wn_s, bl, np.maximum(ab_s, bl),
                            alpha=0.5, color=col, zorder=2,
                            label=f"{name} cm⁻¹  area = {area:.5f}")
            ax.plot(wn_s, bl, color=col, lw=1.2, ls="--", zorder=4)
            peak_idx = np.argmax(ab_s - bl)
            ax.axvline(wn_s[peak_idx], color=col, lw=0.6, ls=":", alpha=0.7)
        ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Absorbance")
        ax.set_title(label); ax.invert_xaxis()
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### Step 4 — The ratio tells you the balance")
    st.markdown(
        "Divide the 1376 area by the 1462 area. "
        "**PP gives a ratio close to or above ~0.6–0.7. "
        "PE gives a ratio much lower** because its 1376 band nearly disappears."
    )
    a1376_pp = get_area(wn_pp, ab_pp, *BAND_1376)
    a1462_pp = get_area(wn_pp, ab_pp, *BAND_1462)
    a1376_pe = get_area(wn_pe, ab_pe, *BAND_1376)
    a1462_pe = get_area(wn_pe, ab_pe, *BAND_1462)
    r_pp = a1376_pp / a1462_pp if a1462_pp > 0 else np.nan
    r_pe = a1376_pe / a1462_pe if a1462_pe > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, label, a76, a62, ratio, color in [
        (axes[0], "90% PP standard", a1376_pp, a1462_pp, r_pp, "#378ADD"),
        (axes[1], "90% PE standard", a1376_pe, a1462_pe, r_pe, "#E05C5C"),
    ]:
        bars = ax.bar(["1376 cm⁻¹\n(PP marker)", "1462 cm⁻¹\n(reference)"],
                      [a76, a62], color=["#7B5EA7", "#E05C5C"],
                      width=0.5, edgecolor="white", linewidth=1.2)
        ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "≈ 0"
        ax.set_title(f"{label}\nRatio = {a76:.4f} ÷ {a62:.4f} = {ratio_str}",
                     fontsize=10)
        ax.set_ylabel("Peak area"); ax.grid(axis="y", alpha=0.2)
        for bar, val in zip(bars, [a76, a62]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.00005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### Step 5 — Where does your dish sit?")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        ex_group = st.selectbox("Group", groups_avail, key="ex_group")
    with ex2:
        ex_dishes = sorted(df[df["group"] == ex_group]["dish_id"].unique(),
                           key=lambda d: (len(d), d))
        ex_dish = st.selectbox("Dish", ex_dishes, key="ex_dish")
    with ex3:
        ex_row = st.selectbox("Row", ROWS, key="ex_row")
        ex_col = st.selectbox("Column", COLS, key="ex_col")

    ex_pos  = f"{ex_row}{ex_col}"
    ex_path = find_csv(ex_dish, ex_pos)

    if ex_path:
        wn_s, ab_s = load_spectrum(ex_path)
        a76_s = get_area(wn_s, ab_s, *BAND_1376)
        a62_s = get_area(wn_s, ab_s, *BAND_1462)
        r_s   = a76_s / a62_s if a62_s > 0 else np.nan

        fig, ax = plt.subplots(figsize=(8, 4))
        labels  = ["90% PP\nstandard", f"Your dish\n{ex_dish} {ex_pos}",
                   "90% PE\nstandard"]
        ratios  = [r_pp, r_s, r_pe if not np.isnan(r_pe) else 0.01]
        colors  = ["#378ADD", "#EF9F27", "#E05C5C"]
        bars = ax.bar(labels, ratios, color=colors, width=0.5,
                      edgecolor="white", linewidth=1.5)
        ax.set_ylabel("1376 / 1462 ratio", fontsize=11)
        ax.set_title("Where does your dish sit?", fontsize=12)
        ax.grid(axis="y", alpha=0.2)
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        ex_data = df[(df["dish_id"] == ex_dish) & (df["grid_pos"] == ex_pos)]
        if not ex_data.empty:
            st.info(
                f"**{ex_dish} position {ex_pos}** has a ratio of "
                f"**{r_s:.4f}** — sitting between the PP standard "
                f"({r_pp:.3f}) and PE standard "
                f"({'~0' if np.isnan(r_pe) else f'{r_pe:.3f}'}), "
                f"consistent with PE contamination in a nominally PP dish."
            )
    else:
        st.info(f"No spectrum found for {ex_dish} {ex_pos}.")
