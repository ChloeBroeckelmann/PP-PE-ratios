"""
Calibration Curves: CH2:CH3 Ratio vs % PE Content
===================================================
Builds TWO separate linear calibration curves:
  1. PP + PE only   (regular polyethylene)
  2. PP + UHMWPE only (ultra-high MW polyethylene)

Three-component mixtures (PP+PE+UHMWPE) are excluded from both.

Outputs:
  calibration_PE.csv / calibration_UHMWPE.csv   — per-sample data
  calibration_curves.png                         — both fits on one plot
  calibration_models.txt                         — slope, intercept, R² for each
  ch2_ch3_ratios_with_PE_estimate.csv            — sample results with both estimates
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

EXCEL_FILE  = "All polymer mixture spectra FTIR _abs.xlsx"
RESULTS_CSV = "ch2_ch3_ratios.csv"
CALIB_PLOT  = "calibration_curves.png"
CALIB_MODEL = "calibration_models.txt"
PATCHED_CSV = "ch2_ch3_ratios_with_PE_estimate.csv"

# Known mixing artefacts — excluded with reason noted in output
OUTLIERS = {
    "60PP_40PE": "granulate mixing artefact (ratio lower than 20% PE point)",
}

# Valley-to-valley windows confirmed by spectral inspection (ASTM D7399-18 bands)
BAND_1376_WINDOW = (1364, 1397)   # CH3 symmetric bend — PP marker
BAND_1462_WINDOW = (1428, 1483)   # CH2 scissoring — PP + PE


# ── Spectrum processing ───────────────────────────────────────────────────────

def local_baseline_area(wn, ab, lo, hi):
    mask = (wn >= lo) & (wn <= hi)
    wn_s, ab_s = wn[mask], ab[mask]
    if len(wn_s) < 3:
        return np.nan
    baseline = np.interp(wn_s, [wn_s[0], wn_s[-1]], [ab_s[0], ab_s[-1]])
    return trapezoid(np.clip(ab_s - baseline, 0, None), wn_s)

def compute_ratio(wn, ab):
    a1376 = local_baseline_area(wn, ab, *BAND_1376_WINDOW)
    a1462 = local_baseline_area(wn, ab, *BAND_1462_WINDOW)
    return a1376 / a1462 if (a1462 and a1462 > 0) else np.nan


# ── Sample classification ─────────────────────────────────────────────────────

def classify_sample(name):
    """
    Returns (pe_pct, curve) where curve is 'PE', 'UHMWPE', or None.
    Only pure two-component PP+PE or PP+UHMWPE blends are accepted.
    """
    # Format A: "PP_PE 60:40", "PP_UHMWPE 30:70", "PE_UHMWPE 90:10"
    m = re.match(
        r'(PP|UHMWPE|PE)[_\s]+(PP|UHMWPE|PE)\s+(\d+)\s*[:\-]\s*(\d+)',
        name, re.IGNORECASE
    )
    if m:
        labels = [m.group(1).upper(), m.group(2).upper()]
        fracs  = [int(m.group(3)), int(m.group(4))]
        total  = sum(fracs)
        non_pp = sum(f for l, f in zip(labels, fracs) if l != 'PP')
        pe_pct = non_pp / total * 100
        has_pp     = 'PP'     in labels
        has_pe     = 'PE'     in labels
        has_uhmwpe = 'UHMWPE' in labels
        if has_pp and has_pe and not has_uhmwpe:
            return pe_pct, 'PE'
        if has_pp and has_uhmwpe and not has_pe:
            return pe_pct, 'UHMWPE'
        return None, 'mixed'

    # Format B: "90PP_10PE", "80PP_20UHMWPE", "1PP_1PE_1UHMWPE..."
    # Match UHMWPE before PE so it takes priority in alternation
    tokens = re.findall(r'(\d+(?:\.\d+)?)(UHMWPE|PP|PE)', name, re.IGNORECASE)
    if tokens:
        labels = [l.upper() for _, l in tokens]
        has_pp     = 'PP'     in labels
        has_pe     = 'PE'     in labels
        has_uhmwpe = 'UHMWPE' in labels
        # Exclude three-component or no-PP blends
        if not has_pp or (has_pe and has_uhmwpe):
            return None, 'mixed'
        total  = sum(float(v) for v, _ in tokens)
        non_pp = sum(float(v) for v, l in tokens if l.upper() != 'PP')
        pe_pct = non_pp / total * 100
        if has_pp and has_pe and not has_uhmwpe:
            return pe_pct, 'PE'
        if has_pp and has_uhmwpe and not has_pe:
            return pe_pct, 'UHMWPE'

    return None, None


# ── Sheet parsers ─────────────────────────────────────────────────────────────

def parse_sheet1(xl):
    df = pd.read_excel(xl, sheet_name='All polymer mixture spectra_01', header=None)
    records = []
    for col in range(0, df.shape[1] - 1, 2):
        name = str(df.iloc[0, col + 1]).strip()
        unit = str(df.iloc[1, col + 1]).strip().lower()
        if 'transmittance' in unit:
            continue
        try:
            wn = pd.to_numeric(df.iloc[2:, col],     errors='coerce').dropna().values
            ab = pd.to_numeric(df.iloc[2:, col + 1], errors='coerce').dropna().values
            n  = min(len(wn), len(ab))
            wn, ab = wn[:n], ab[:n]
            idx = np.argsort(wn)
            records.append({'name': name, 'wn': wn[idx], 'ab': ab[idx]})
        except Exception:
            continue
    return records

def parse_sheet2(xl):
    df = pd.read_excel(xl, sheet_name='All polymer mixture spectra FTI', header=None)
    wn_row = pd.to_numeric(df.iloc[0, 2:], errors='coerce').values
    records = []
    for row in range(1, df.shape[0]):
        name = str(df.iloc[row, 0]).strip()
        if not name or name.lower() == 'nan':
            continue
        ab   = pd.to_numeric(df.iloc[row, 2:], errors='coerce').values
        n    = min(len(wn_row), len(ab))
        wn_s, ab_s = wn_row[:n], ab[:n]
        mask = ~np.isnan(wn_s) & ~np.isnan(ab_s)
        wn_s, ab_s = wn_s[mask], ab_s[mask]
        records.append({'name': name, 'wn': wn_s[np.argsort(wn_s)],
                        'ab': ab_s[np.argsort(wn_s)]})
    return records


# ── Fit + plot helpers (quadratic) ───────────────────────────────────────────

def fit_curve(calib_df):
    """Quadratic fit: ratio = a·x² + b·x + c. Returns coeffs and R²."""
    x = calib_df['pe_pct'].values
    y = calib_df['ch2_ch3_ratio'].values
    coeffs = np.polyfit(x, y, deg=2)          # [a, b, c]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2     = 1 - ss_res / ss_tot
    return coeffs, r2

def predict(coeffs, x_arr):
    return np.polyval(coeffs, x_arr)

def invert_quadratic(coeffs, ratio_val):
    """Solve a·x² + b·x + (c - ratio) = 0; return the root in [0,100]."""
    a, b, c = coeffs
    roots = np.roots([a, b, c - ratio_val])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and 0 <= r.real <= 100]
    return round(real_roots[0], 1) if real_roots else np.nan


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    xl      = pd.ExcelFile(EXCEL_FILE)
    spectra = parse_sheet1(xl) + parse_sheet2(xl)
    print(f'Loaded {len(spectra)} absorbance spectra.')

    rows_pe, rows_uh = [], []

    for s in spectra:
        pe_pct, curve = classify_sample(s['name'])
        if curve not in ('PE', 'UHMWPE'):
            continue
        ratio = compute_ratio(s['wn'], s['ab'])
        if np.isnan(ratio):
            continue
        row = {'sample': s['name'], 'pe_pct': round(pe_pct, 1),
               'ch2_ch3_ratio': round(ratio, 4)}
        (rows_pe if curve == 'PE' else rows_uh).append(row)

    calib_pe = pd.DataFrame(rows_pe).drop_duplicates().sort_values('pe_pct')
    calib_uh = pd.DataFrame(rows_uh).drop_duplicates().sort_values('pe_pct')

    # Flag and remove known mixing artefacts
    for name, reason in OUTLIERS.items():
        if name in calib_pe['sample'].values:
            print(f'\n  Outlier excluded from PE curve: {name} ({reason})')
            calib_pe = calib_pe[calib_pe['sample'] != name]
        if name in calib_uh['sample'].values:
            print(f'\n  Outlier excluded from UHMWPE curve: {name} ({reason})')
            calib_uh = calib_uh[calib_uh['sample'] != name]

    calib_pe.to_csv('calibration_PE.csv',     index=False)
    calib_uh.to_csv('calibration_UHMWPE.csv', index=False)
    print(f'\nPP+PE curve:     {len(calib_pe)} samples (quadratic fit)')
    print(f'PP+UHMWPE curve: {len(calib_uh)} samples (quadratic fit)')

    coeffs_pe, r2_pe = fit_curve(calib_pe)
    coeffs_uh, r2_uh = fit_curve(calib_uh)
    a_pe, b_pe, c_pe = coeffs_pe
    a_uh, b_uh, c_uh = coeffs_uh

    print(f'\nPP+PE    fit: ratio = {a_pe:.5f}x² + {b_pe:.5f}x + {c_pe:.4f}   R²={r2_pe:.3f}')
    print(f'PP+UHMWPE fit: ratio = {a_uh:.5f}x² + {b_uh:.5f}x + {c_uh:.4f}   R²={r2_uh:.3f}')

    with open(CALIB_MODEL, 'w') as f:
        f.write('[PE]\n')
        f.write(f'a={a_pe}\nb={b_pe}\nc={c_pe}\nr2={r2_pe}\n\n')
        f.write('[UHMWPE]\n')
        f.write(f'a={a_uh}\nb={b_uh}\nc={c_uh}\nr2={r2_uh}\n')
    print(f'Models saved to {CALIB_MODEL}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    x_fit   = np.linspace(0, 90, 300)
    results = pd.read_csv(RESULTS_CSV)
    sample_ratios = results['ch2_ch3_ratio'].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, calib, coeffs, r2, label, color in [
        (axes[0], calib_pe, coeffs_pe, r2_pe, 'PP + PE (regular)',  '#E05C5C'),
        (axes[1], calib_uh, coeffs_uh, r2_uh, 'PP + UHMWPE',        '#7B5EA7'),
    ]:
        x, y  = calib['pe_pct'].values, calib['ch2_ch3_ratio'].values
        y_fit = predict(coeffs, x_fit)
        a, b, c = coeffs

        ax.scatter(x, y, color='#333', s=40, zorder=4, label='Standards')
        ax.plot(x_fit, y_fit, color=color, lw=2,
                label=f'ratio = {a:.4f}x² + {b:.4f}x + {c:.3f}\nR² = {r2:.3f}')
        ax.axhspan(sample_ratios.min(), sample_ratios.max(),
                   alpha=0.08, color='#378ADD',
                   label=f'Your samples ({sample_ratios.min():.2f}–{sample_ratios.max():.2f})')
        ax.set_xlabel('PE content (% by weight)', fontsize=11)
        ax.set_ylabel('CH₂:CH₃ ratio', fontsize=11)
        ax.set_title(f'Calibration: {label}', fontsize=12)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.2)

    plt.suptitle('CH₂:CH₃ ratio calibration curves (quadratic fit)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(CALIB_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {CALIB_PLOT}')

    # ── Patch results CSV ─────────────────────────────────────────────────────
    results['pe_pct_PE_curve'] = results['ch2_ch3_ratio'].apply(
        lambda r: invert_quadratic(coeffs_pe, r) if not np.isnan(r) else np.nan)
    results['pe_pct_UHMWPE_curve'] = results['ch2_ch3_ratio'].apply(
        lambda r: invert_quadratic(coeffs_uh, r) if not np.isnan(r) else np.nan)
    results.to_csv(PATCHED_CSV, index=False)
    print(f'\nResults with PE estimates saved to {PATCHED_CSV}')

    print('\nMean estimated %PE by group (PE curve | UHMWPE curve):')
    grp = results.groupby('group')[['pe_pct_PE_curve', 'pe_pct_UHMWPE_curve']].mean().round(1)
    print(grp.to_string())


if __name__ == '__main__':
    main()
