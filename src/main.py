import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, normal_ad
from scipy.stats import normaltest
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning
warnings.filterwarnings("ignore", category=SMValueWarning,
                        message="A date index has been provided, but it has no associated frequency")


# =========================
# 1) Load Data
# =========================

files = {
    "arbitrum_burns": "ARBburns.csv",
    "arbitrum_mints": "ARBmints.csv",
    "arbitrum_swaps": "ARBswaps.csv",
    "ethereum_burns": "ETHburns.csv",
    "ethereum_mints": "ETHmints.csv",
    "ethereum_swaps": "ETHswaps.csv",
    "polygon_burns": "POLYburns.csv",
    "polygon_mints": "POLYmints.csv",
    "polygon_swaps": "POLYswaps.csv"
}

data = {name: pd.read_csv(path) for name, path in files.items()}

if len(data) == len(files):
    print("All files loaded.")
else:
    print(f"Warning: Only {len(data)} of {len(files)} files loaded.")


# =========================
# 2) Helpers to extract and standardize prices from swaps, and save tables,figures with captions
# =========================

import os, shutil, textwrap
from datetime import timedelta
import re

def _latex_escape(obj):
    s = str(obj)
    # escape LaTeX special chars
    repl = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
               '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'
    }
    return ''.join(repl.get(ch, ch) for ch in s)

def _fmt_cell(v, floatfmt):
    if pd.isna(v): return ''
    if isinstance(v, (np.floating, float)):
        if isinstance(floatfmt, str):
            try: return floatfmt % float(v)
            except: return f"{float(v):.3f}"
        else:
            return f"{float(v):.3f}"
    return _latex_escape(v)

def _simple_tabular(df, index=True, floatfmt="%.3f"):
    # Make a safe copy with nicer datetimes
    df2 = df.copy()
    for c in df2.columns:
        if np.issubdtype(df2[c].dtype, np.datetime64):
            df2[c] = df2[c].dt.strftime("%Y-%m-%d %H:%M")
    if isinstance(df2.index, pd.DatetimeIndex):
        df2 = df2.copy()
        df2.index = df2.index.strftime("%Y-%m-%d %H:%M")

    cols = list(df2.columns)
    # Column alignment (plain \hline so no booktabs needed)
    ncols = len(cols) + (1 if index else 0)
    align = 'l' * ncols

    header_cells = ([""] if index else []) + [_latex_escape(c) for c in cols]
    lines = []
    lines.append(r"\begin{tabular}{" + align + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    for r in range(len(df2)):
        row = []
        if index:
            row.append(_latex_escape(df2.index[r]))
        for c in cols:
            row.append(_fmt_cell(df2.iloc[r][c], floatfmt))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)

def _wrap_table_latex(df, caption, label, index=True, floatfmt="%.3f"):
    body = _simple_tabular(df if index else df.reset_index(drop=True),
                           index=index, floatfmt=floatfmt)
    return f"""% Auto-generated (no-Jinja)
\\begin{{table}}[H]
\\centering
{body}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""

OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)

def _wrap_table_latex(df, caption, label, index=True, floatfmt="%.3f"):
    body = df.to_latex(index=index, float_format=(lambda x: floatfmt % x) if isinstance(floatfmt, str) else None)
    return textwrap.dedent(f"""\
    % Auto-generated
    \\begin{{table}}[H]
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    {body}
    \\end{{table}}
    """)

def save_table(df, base, caption, label, index=True, floatfmt="%.3f"):
    """
    Save a CSV and a LaTeX snippet (with caption+label) into ./artifacts
    WITHOUT using DataFrame.style (so no Jinja2 needed).
    """
    OUT.mkdir(exist_ok=True)
    base = str(base)

    # 1) CSV (UTF-8)
    (OUT / f"{base}.csv").write_text(
        df.to_csv(index=index, encoding="utf-8"), encoding="utf-8"
    )  # write via string so Windows newline handling is consistent

    # 2) LaTeX table (UTF-8) using pandas.to_latex (no Jinja)
    tex_body = _wrap_table_latex(df, caption=caption, label=label,
                                 index=index, floatfmt=floatfmt)
    (OUT / f"{base}.tex").write_text(tex_body, encoding="utf-8")

    # 3) Plain caption text (handy for your doc tooling)
    (OUT / f"{base}.caption.txt").write_text(caption, encoding="utf-8")

def save_caption_snippet_for_image(filename_png, base, caption, label):
    tex = f"""% Auto-generated
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.95\\linewidth]{{{_latex_escape(filename_png)}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""
    (OUT / f"{base}.tex").write_text(tex, encoding="utf-8")
    (OUT / f"{base}.caption.txt").write_text(caption, encoding="utf-8")

def save_figure(fig, base, caption, label):
    """Saves PNG/SVG under artifacts/ and a LaTeX snippet with caption+label."""
    png = OUT / f"{base}.png"
    svg = OUT / f"{base}.svg"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(svg, bbox_inches="tight")
    save_caption_snippet_for_image(png.name, base, caption, label)

def ecdf(y):
    """Return x(sorted), F(x) for ECDF."""
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    y = np.sort(y)
    if y.size == 0:
        return y, y
    F = np.arange(1, y.size+1) / y.size
    return y, F

def pick_windows_for_spreads(spreads_df, days=7):
    """
    Choose 'calm' and 'stress' 7-day windows based on rolling std of |spreads|.
    Returns (calm_start, calm_end, stress_start, stress_end).
    """
    s = spreads_df.copy()
    s = s[['ARB_vs_ETH_bps','POLY_vs_ETH_bps']].dropna(how='any')
    if s.empty:
        return None, None, None, None
    mag = s.abs().sum(axis=1)
    roll = mag.rolling(f'{days}D', min_periods=60).std()
    # windows keyed by window end; select min and max
    calm_end = roll.idxmin()
    stress_end = roll.idxmax()
    calm_start  = calm_end - timedelta(days=days) if pd.notna(calm_end) else None
    stress_start = stress_end - timedelta(days=days) if pd.notna(stress_end) else None
    return calm_start, calm_end, stress_start, stress_end

def to_datetime_best(df):
    """Coerce a likely timestamp column into pandas datetime (UTC, no tz)."""
    for cand in ['timestamp', 'time', 'block_timestamp', 'blockTime', 'block_time', 'evt_block_time']:
        if cand in df.columns:
            try:
                ts = pd.to_datetime(df[cand], utc=True, errors='coerce')
                if ts.notna().any():
                    return ts.dt.tz_localize(None)
            except Exception:
                pass
    # if nothing found, return a monotonically increasing index as time (last resort)
    return pd.to_datetime(pd.Series(range(len(df))), unit='s')

def _sqrtPriceX96_to_price_usdc_per_eth(sqrtpx96_series, token0_is_usdc_assumption=True):
    """
    Uniswap v3 convention: sqrtPriceX96 = sqrt(price_token1 / price_token0) in Q64.96.
    For canonical ETH/USDC pools, token0 is USDC and token1 is WETH/ETH.
    - If token0 == USDC, token1 == ETH -> sqrt = sqrt(ETH/USDC); we want USDC/ETH -> inverse square.
    - If token0 == ETH, token1 == USDC -> sqrt = sqrt(USDC/ETH); we want USDC/ETH -> square.
    """
    x = sqrtpx96_series.astype(float) / (2**96)
    if token0_is_usdc_assumption:
        return (1.0 / (x**2)).replace({np.inf: np.nan})
    else:
        return (x**2)

def best_price_from_swaps(df):
    """
    Try multiple ways to get ETH price in USDC from a generic swaps export.
    We prefer a direct 'price' if present and plausible; otherwise fall back to sqrtPriceX96 or trade ratio.
    Chooses the candidate whose median lives in a reasonable ETH/USD band (100..100000).
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype='datetime64[ns]')

    # Normalize column names (lowercase) for robust access
    cols = {c: c.lower() for c in df.columns}
    dfl = df.rename(columns=cols).copy()

    t = to_datetime_best(dfl)

    candidates = {}

    # 1.1a) Direct 'price' column (various common names)
    for cname in ['price', 'midprice', 'mid_price', 'execution_price', 'p']:
        if cname in dfl.columns:
            candidates['direct_price'] = pd.to_numeric(dfl[cname], errors='coerce')

    # 1.1b) Price from tick (Uniswap v3 convention)
    if 'tick' in dfl.columns:
        candidates['price_from_tick'] = 1.0001 ** pd.to_numeric(dfl['tick'], errors='coerce')

    # 1.2) From sqrtPriceX96 (try both assumptions for token0)
    if 'sqrtpricex96' in dfl.columns:
        sp = pd.to_numeric(dfl['sqrtpricex96'], errors='coerce')
        candidates['sqrt_token0_is_usdc'] = _sqrtPriceX96_to_price_usdc_per_eth(sp, token0_is_usdc_assumption=True)
        candidates['sqrt_token0_is_eth']  = _sqrtPriceX96_to_price_usdc_per_eth(sp, token0_is_usdc_assumption=False)

    # 1.3) From trade legs (amounts). Use |amount1|/|amount0| and |amount0|/|amount1| as two hypotheses.
    # Typical Uniswap exports: amount0 (token0 change), amount1 (token1 change).
    # We try both directions and pick the plausible one.
    if 'amount0' in dfl.columns and 'amount1' in dfl.columns:
        a0 = pd.to_numeric(dfl['amount0'], errors='coerce')
        a1 = pd.to_numeric(dfl['amount1'], errors='coerce')
        with np.errstate(divide='ignore', invalid='ignore'):
            r10 = (a1.abs() / a0.abs()).replace({np.inf: np.nan})
            r01 = (a0.abs() / a1.abs()).replace({np.inf: np.nan})
        candidates['ratio_a1_over_a0'] = r10
        candidates['ratio_a0_over_a1'] = r01

    # Pick the candidate whose median is in a plausible ETH/USD band
    plaus_low, plaus_high = 100.0, 100000.0
    best_key, best_med = None, None
    for k, s in candidates.items():
        med = s.dropna().median()
        if med is not None and np.isfinite(med) and plaus_low <= med <= plaus_high:
            if best_med is None:
                best_key, best_med = k, med

    if best_key is None:
        # Fall back: pick the candidate with the most non-nan and finite observations
        best_key = max(candidates.keys(), key=lambda k: candidates[k].replace([np.inf, -np.inf], np.nan).notna().sum()) if candidates else None

    if best_key is None:
        # Give back an empty price series
        return pd.Series(dtype=float), t

    px = candidates[best_key].astype(float)
    return px, t

def chain_minute_series(swaps_df):
    """Return a clean 1-minute ETH/USDC price series from a chain's swaps dataframe."""
    px, t = best_price_from_swaps(swaps_df)
    if px.empty:
        return pd.Series(dtype=float)
    s = pd.Series(px.values, index=t).sort_index()
    # Basic outlier clamp (5×MAD around rolling median) to stabilize resampling
    med = s.rolling(50, min_periods=10).median()
    mad = (s - med).abs().rolling(50, min_periods=10).median()
    s_clamped = s.where((mad == 0) | ((s - med).abs() <= 5 * mad), np.nan)
    # Resample to 1min using last observed price in the minute, then ffill
    s_1m = s_clamped.resample('1min').last().ffill()
    return s_1m.rename('price_usdc_per_eth')

# =========================
# 3) Build per-chain 1-minute prices
# =========================

px_eth_1m = chain_minute_series(data['ethereum_swaps'])
px_arb_1m = chain_minute_series(data['arbitrum_swaps'])
px_poly_1m = chain_minute_series(data['polygon_swaps'])

# Align into one DataFrame
prices_1m = pd.concat(
    [px_eth_1m.rename('ETH_mainnet'),
     px_arb_1m.rename('ARB'),
     px_poly_1m.rename('POLY')],
    axis=1
).sort_index()

# Optional trimming to common intersection so spreads/arbs are apples-to-apples
common = prices_1m.dropna(how='any')

# Print prices extracted and standardized successfully
print("Prices extracted and standardized successfully (sample):")
print(prices_1m.dropna(how='any').head(10))
print(f"\nCounts (non-NaN minutes): ETH={prices_1m['ETH_mainnet'].notna().sum()}, "
      f"ARB={prices_1m['ARB'].notna().sum()}, POLY={prices_1m['POLY'].notna().sum()}")

# Print 1 minute Price series built
print("\n1-minute price series built (common intersection, first 10 rows):")
print(common.head(10))

# =========================
# 4) Compute ETH-anchored spreads (bps) vs Ethereum
# =========================

def spread_bps(child, anchor):
    return (child/anchor - 1.0) * 10_000.0

spreads = pd.DataFrame(index=common.index)
spreads['ARB_vs_ETH_bps']  = spread_bps(common['ARB'],  common['ETH_mainnet'])
spreads['POLY_vs_ETH_bps'] = spread_bps(common['POLY'], common['ETH_mainnet'])

# After computing `spreads` the first time:
if 'ARB_vs_ETH_bps' in spreads.columns:
    med_abs = spreads['ARB_vs_ETH_bps'].abs().median()
    if np.isfinite(med_abs) and med_abs > 5000:  # near 10,000 bps => inverted price
        print("ARB price likely inverted vs ETH anchor; correcting ARB series...")
        prices_1m['ARB'] = 1.0 / prices_1m['ARB']
        # Rebuild intersection & spreads consistently
        common = prices_1m.dropna(how='any')
        spreads = pd.DataFrame(index=common.index)
        spreads['ARB_vs_ETH_bps']  = (common['ARB']/common['ETH_mainnet'] - 1.0)*10_000.0
        spreads['POLY_vs_ETH_bps'] = (common['POLY']/common['ETH_mainnet'] - 1.0)*10_000.0


# Print Spreads with ETH as anchor computed
print("\nSpreads (bps) with ETH as anchor (first 10 rows):")
print(spreads.head(10))

# =========================
# 5) Detect arbitrage signals
# =========================
# Conservative thresholds: 10 bps for ARB, 15 bps for POLY
thr = {'ARB_vs_ETH_bps': 10.0, 'POLY_vs_ETH_bps': 15.0}

signals = []
for name in ['ARB_vs_ETH_bps', 'POLY_vs_ETH_bps']:
    s = spreads[name].dropna()
    up   = s[s >  thr[name]]
    down = s[s < -thr[name]]
    if not up.empty:
        signals.append(
            pd.DataFrame({
                'pair'     : name,
                'direction': 'L2 overpriced vs ETH (sell L2 / buy ETH)',
                'spread_bps': up
            })
        )
    if not down.empty:
        signals.append(
            pd.DataFrame({
                'pair'     : name,
                'direction': 'L2 underpriced vs ETH (buy L2 / sell ETH)',
                'spread_bps': down
            })
        )

arb_signals = pd.concat(signals).sort_values('spread_bps', key=lambda x: x.abs(), ascending=False) if signals else pd.DataFrame(columns=['pair','direction','spread_bps'])

# Add the contemporaneous prices for context
if not arb_signals.empty:
    snap = prices_1m[['ETH_mainnet','ARB','POLY']].reindex(arb_signals['spread_bps'].index)
    arb_report = arb_signals.join(snap)
else:
    arb_report = pd.DataFrame(columns=['pair','direction','spread_bps','ETH_mainnet','ARB','POLY'])

# =========================
# 6) Post-signal convergence metrics (fast + robust)
# =========================

# --- choose the signals source (prefer the enriched report with prices) ---
if 'arb_report' in globals() and isinstance(arb_report, pd.DataFrame) and not arb_report.empty:
    _signals = arb_report.copy()
elif 'arb_signals' in globals() and isinstance(arb_signals, pd.DataFrame) and not arb_signals.empty:
    _signals = arb_signals.copy()
else:
    raise RuntimeError("No arbitrage signals found. Ensure Step 5 produced 'arb_report' or 'arb_signals'.")

# --- ensure a 'ts' column exists and is datetime (robust) ---
def ensure_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    print("_signals columns:", _signals.columns)
    if "ts" in _signals.columns:
        if not np.issubdtype(_signals["ts"].dtype, np.datetime64):
            print("Column 'ts' is not datetime64 type.")
    if 'ts' in g.columns:
        g['ts'] = pd.to_datetime(g['ts'], utc=True, errors='coerce').dt.tz_localize(None)
        return g
    if isinstance(g.index, pd.DatetimeIndex):
        g = g.reset_index()
        first_col = g.columns[0]
        print("Column 'ts' not found in _signals DataFrame.")
        g = g.rename(columns={first_col: 'ts'})
        g['ts'] = pd.to_datetime(g['ts'], utc=True, errors='coerce').dt.tz_localize(None)
        return g
    cand_names = tuple(n.lower() for n in ('ts','timestamp','time','datetime','date','block_timestamp','blockTime','block_time'))
    ts_col = next((c for c in g.columns if str(c).lower() in cand_names), None)
    if ts_col is not None:
        g = g.rename(columns={ts_col: 'ts'})
        g['ts'] = pd.to_datetime(g['ts'], utc=True, errors='coerce').dt.tz_localize(None)
        return g
    for fb in ('index','level_0'):
        if fb in g.columns:
            g = g.rename(columns={fb: 'ts'})
            g['ts'] = pd.to_datetime(g['ts'], utc=True, errors='coerce').dt.tz_localize(None)
            return g
    raise RuntimeError("Could not create a 'ts' column for signals.")

_signals = ensure_ts_column(_signals)

# --- derive 'market' if missing, using the 'pair' naming from Step 5 ---
if 'market' not in _signals.columns:
    if 'pair' in _signals.columns:
        _signals['market'] = np.where(
            _signals['pair'].str.contains('ARB', case=False), 'arbitrum',
            np.where(_signals['pair'].str.contains('POLY', case=False), 'polygon', 'unknown')
        )
    else:
        _signals['market'] = 'unknown'

# --- fill per-row band in bps if not present (use your thresholds by pair; otherwise use |spread| at signal) ---
if 'band_bp' not in _signals.columns:
    def _band(row):
        if 'pair' in _signals.columns and isinstance(row.get('pair', None), str):
            return thr.get(row['pair'], np.nan)
        return np.nan
    _signals['band_bp'] = _signals.apply(_band, axis=1)
    if 'spread_bps' in _signals.columns:
        _signals['band_bp'] = _signals['band_bp'].fillna(_signals['spread_bps'].abs())
    else:
        _signals['band_bp'] = _signals['band_bp'].fillna(0.0)

# --- spreads frame from Step 4 on a 1-min grid ---
if 'spreads' not in globals():
    raise RuntimeError("Expected 'spreads' from Step 4 is missing.")

_spreads = spreads.copy()
_spreads.index = pd.to_datetime(_spreads.index, utc=True, errors='coerce').tz_localize(None)
_spreads = _spreads.asfreq('min')  # fix deprecation: 'T' -> 'min'

# pick spread columns for ARB and POLY vs ETH
col_arb  = next((c for c in _spreads.columns if 'arb'  in c.lower()), None)
col_poly = next((c for c in _spreads.columns if 'poly' in c.lower()), None)
if col_arb is None and col_poly is None:
    raise RuntimeError("Could not locate ARB/POLY spread columns in 'spreads'.")

# --- precompute arrays for speed ---
idx = _spreads.index
s_arb_arr  = _spreads[col_arb].astype(float).to_numpy()  if col_arb  is not None else None
s_poly_arr = _spreads[col_poly].astype(float).to_numpy() if col_poly is not None else None
n = len(idx)

# normalize signals to minute timestamps and within index bounds
_signals['ts_min'] = _signals['ts'].dt.floor('min')
mask_in = _signals['ts_min'].between(idx[0], idx[-1])
_signals = _signals.loc[mask_in].copy()
if _signals.empty:
    raise RuntimeError("All signals fell outside the spreads index range; check your time alignment.")

# vectorized nearest index for all signals once (avoid per-row .reindex(method='nearest'))
i0_all = idx.get_indexer(_signals['ts_min'], method='nearest')  # numpy array of positions
_signals['i0'] = i0_all

def _choose_series(mkt):
    m = str(mkt).lower()
    if ('arb' in m) or ('arbitrum' in m):
        return s_arb_arr
    if ('poly' in m) or ('polygon' in m):
        return s_poly_arr
    return None

def _metrics_from_array(s_arr, i0, band_bp, max_lookahead=120):
    """Compute convergence metrics from a numpy series s_arr using integer positions only."""
    if s_arr is None or not np.isfinite(i0):
        return np.nan, np.nan, False, np.nan, np.nan
    i0 = int(i0)
    if i0 < 0 or i0 >= len(s_arr):
        return np.nan, np.nan, False, np.nan, np.nan
    end = min(len(s_arr), i0 + max_lookahead + 1)
    seg = s_arr[i0:end]
    if seg.size == 0 or not np.isfinite(seg[0]):
        return np.nan, np.nan, False, np.nan, np.nan

    s0 = float(seg[0])
    sign0 = np.sign(s0) if s0 != 0 else 0
    abs_seg = np.abs(seg)
    peak_abs = float(np.nanmax(abs_seg))
    # sum in bp-min; NaNs treated as 0 to avoid killing the sum
    auc = float(np.nansum(np.where(np.isfinite(abs_seg), abs_seg, 0.0)))

    cond_close = abs_seg <= band_bp
    cond_flip  = (sign0 != 0) & (np.sign(seg) == -sign0)
    cond = cond_close | cond_flip
    if cond.any():
        minutes_to_close = int(np.argmax(cond))  # first True
        closed = True
    else:
        minutes_to_close = np.nan
        closed = False
    return s0, minutes_to_close, closed, peak_abs, auc

records = []
# keep only valid integer positions inside the spreads index
_signals = _signals.dropna(subset=['i0']).copy()
_signals['i0'] = _signals['i0'].astype('int64')
n = len(idx)
_signals = _signals[(_signals['i0'] >= 0) & (_signals['i0'] < n)]

# optional: de-dup same minute & market to reduce work if many signals cluster
_signals = _signals.drop_duplicates(subset=['ts_min', 'market'])

# sort for nicer output order
_signals = _signals.sort_values(['ts_min', 'market'])

# iterate with itertuples (much faster than iterrows)
for ts_min, market, i0, band in _signals[['ts_min','market','i0','band_bp']].itertuples(index=False, name=None):
    s_arr = _choose_series(market)
    if s_arr is None:
        # pick whichever spread has larger |value| at i0 across available series
        cand = []
        if s_arb_arr is not None:
            cand.append((s_arb_arr, 'arbitrum'))
        if s_poly_arr is not None:
            cand.append((s_poly_arr, 'polygon'))
        if not cand:
            continue
        vals = [(arr, name, arr[i0] if 0 <= i0 < len(arr) else np.nan) for arr, name in cand]
        vals = [(arr, name, v) for arr, name, v in vals if np.isfinite(v)]
        if not vals:
            continue
        s_arr, market = max(vals, key=lambda t: abs(t[2]))[:2]

    s0, tt, closed, peak_abs, auc = _metrics_from_array(s_arr, i0, float(band))
    if not np.isfinite(s0):
        continue

    records.append({
        'ts'              : ts_min,
        'market'          : market,
        'spread0_bp'      : float(s0),
        'band_bp'         : float(band),
        'minutes_to_close': tt,
        'closed'          : bool(closed),
        'peak_abs_bp'     : float(peak_abs),
        'auc_bpmin'       : float(auc)
    })

convergence = pd.DataFrame.from_records(records).sort_values(['ts','market']).reset_index(drop=True)

# --- Print latency & convergence measurement results ---
print("\n=== Latency & Convergence: Overview ===")
total = len(convergence)
closed = int(convergence['closed'].sum()) if total else 0
open_  = total - closed
print(f"Signals processed: {total}")
print(f"Closed (within band or sign flip): {closed}")
print(f"Still open after lookahead: {open_}")

if total and closed:
    closed_df = convergence[convergence['closed']].copy()

    # Overall time-to-close stats (closed only)
    def q(x, p): 
        try: 
            return float(x.quantile(p)) 
        except Exception: 
            return np.nan

    med = float(closed_df['minutes_to_close'].median())
    mean = float(closed_df['minutes_to_close'].mean())
    p25 = q(closed_df['minutes_to_close'], 0.25)
    p75 = q(closed_df['minutes_to_close'], 0.75)
    mn  = float(closed_df['minutes_to_close'].min())
    mx  = float(closed_df['minutes_to_close'].max())

    print("\n=== Time-to-close (minutes) — Overall (closed only) ===")
    print(f"median={med:.2f} | mean={mean:.2f} | p25={p25:.2f} | p75={p75:.2f} | min={mn:.0f} | max={mx:.0f}")

    # By-market summary (closed only)
    by = (closed_df.groupby('market')['minutes_to_close']
          .agg(['count','median','mean','min','max', 
                lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
          .rename(columns={'<lambda_0>':'p25','<lambda_1>':'p75'}))
    print("\n=== Time-to-close (minutes) — By Market (closed only) ===")
    print(by.to_string(float_format=lambda x: f"{x:,.2f}"))

# Top events: slowest to close (including those that never closed -> show at end)
if total:
    # Put non-closed at the bottom by using large placeholder
    tt = convergence.copy()
    tt['_tt_sort'] = np.where(tt['closed'], tt['minutes_to_close'], np.inf)
    slowest = tt.sort_values(['_tt_sort','ts','market']).head(10).drop(columns=['_tt_sort'])
    print("\n=== Slowest closures (top 10) ===")
    if slowest.empty:
        print("No signals to report.")
    else:
        print(slowest[['ts','market','spread0_bp','band_bp','minutes_to_close','closed','peak_abs_bp','auc_bpmin']]
              .to_string(index=False, 
                         formatters={'spread0_bp':'{:.2f}'.format,
                                     'band_bp':'{:.2f}'.format,
                                     'minutes_to_close': (lambda x: 'NA' if pd.isna(x) or np.isinf(x) else f'{int(x)}'),
                                     'peak_abs_bp':'{:.2f}'.format,
                                     'auc_bpmin':'{:.2f}'.format}))

    # Largest initial mispricings
    biggest = convergence.reindex(
        convergence['spread0_bp'].abs().sort_values(ascending=False).index
    ).head(10)
    print("\n=== Largest initial mispricings |spread0_bp| (top 10) ===")
    if biggest.empty:
        print("No signals to report.")
    else:
        print(biggest[['ts','market','spread0_bp','band_bp','minutes_to_close','closed','peak_abs_bp','auc_bpmin']]
              .to_string(index=False, 
                         formatters={'spread0_bp':'{:.2f}'.format,
                                     'band_bp':'{:.2f}'.format,
                                     'minutes_to_close': (lambda x: 'NA' if pd.isna(x) else f'{int(x)}'),
                                     'peak_abs_bp':'{:.2f}'.format,
                                     'auc_bpmin':'{:.2f}'.format}))
else:
    print("\nNo convergence records to summarize.")

# =========================
# 7) Recursive SVAR via VAR + Cholesky (trades-first identification)
# =========================

# 7.1 Build inputs on a strict 1-min grid
rets = np.log(common[['ETH_mainnet','ARB','POLY']]).diff()
rets.columns = ['R_ETH','R_ARB','R_POLY']

# If OF_* already exist from earlier step, use them; else build quickly here
def _ensure_of(name, swaps_df, px, label):
    if name in globals():
        return globals()[name]
    # lightweight fallback (same sign convention: buys positive)
    df = swaps_df.copy()
    df.columns = [c.lower() for c in df.columns]
    t = None
    for cand in ['timestamp','time','block_timestamp','blocktime','block_time','evt_block_time']:
        if cand in df.columns:
            t = pd.to_datetime(df[cand], utc=True, errors='coerce').dt.tz_localize(None)
            break
    if t is None:
        t = pd.to_datetime(pd.Series(range(len(df))), unit='s')
    of_raw = None
    for eth_col in ['amount1','delta_y','amount1_in','amount1_out']:
        if eth_col in df.columns:
            of_raw = -pd.to_numeric(df[eth_col], errors='coerce'); break
    if of_raw is None and 'amount1' in df.columns:
        of_raw = -pd.to_numeric(df['amount1'], errors='coerce')
    if of_raw is None:
        return pd.Series(dtype=float, name=f'OF_{label}')
    s = pd.Series(of_raw.values, index=t).sort_index().resample('min').sum(min_count=1)
    px1m = px.reindex(s.index).ffill()
    of_usd = s * px1m
    mu = of_usd.rolling(60, min_periods=12).mean()
    sd = of_usd.rolling(60, min_periods=12).std(ddof=0)
    return ((of_usd - mu) / sd).rename(f'OF_{label}')

OF_ETH  = _ensure_of('OF_ETH',  data['ethereum_swaps'], prices_1m['ETH_mainnet'], 'ETH')
OF_ARB  = _ensure_of('OF_ARB',  data['arbitrum_swaps'], prices_1m['ARB'],         'ARB')
OF_POLY = _ensure_of('OF_POLY', data['polygon_swaps'],  prices_1m['POLY'],        'POLY')

Y = pd.concat([OF_ETH, OF_ARB, OF_POLY, rets], axis=1).asfreq('min').dropna()
Y.index = Y.index.tz_localize(None)
Y = Y[['OF_ETH','OF_ARB','OF_POLY','R_ETH','R_ARB','R_POLY']]  # explicit order (trades -> returns)

print("\n=== Recursive SVAR (VAR+Cholesky) input (head) ===")
print(Y.head())
print("Samples:", len(Y))

# 7.2 Choose lag by AIC on reduced-form VAR (conservative for minute data)
maxlags = min(6, max(2, len(Y)//200))
aic = {}
for p in range(1, maxlags+1):
    try:
        aic[p] = VAR(Y).fit(p).aic
    except Exception:
        pass
p_star = min(aic, key=aic.get) if aic else 2
print(f"Recursive SVAR chosen lag p = {p_star}")

# 7.3 Fit VAR and get Cholesky-based structural shocks (trades-first ordering)
var = VAR(Y).fit(p_star)
Sigma_u = var.sigma_u
P = np.linalg.cholesky(Sigma_u)  # lower-triangular impact matrix (contemporaneous)
impact = pd.DataFrame(P, index=Y.columns, columns=Y.columns)
print("\nContemporaneous impact (Cholesky, lower-triangular):")
print(impact.round(4).to_string())

# 7.4 Orthogonalized IRFs (structural) — horizon 30 minutes
H = 30
irf = var.irf(H)   # already orthogonalized via Cholesky using the column order of Y

print("\n=== IRF snapshot: Returns’ response to Trade shocks (horizon 30) ===")
for shock in ['OF_ETH','OF_ARB','OF_POLY']:
    j = list(Y.columns).index(shock)
    print(f"\nShock: {shock}")
    for resp in ['R_ETH','R_ARB','R_POLY']:
        i = list(Y.columns).index(resp)
        path = irf.irfs[:, i, j]
        first = float(path[0])   # contemporaneous (t=0)
        one   = float(path[1])   # next minute (t=1)
        peak_idx = int(np.nanargmax(np.abs(path)))
        peak_val = float(path[peak_idx])
        print(f"{resp}: t=0={first:+.4e}, t=1={one:+.4e}, peak@t={peak_idx}={peak_val:+.4e}")

# 7.5 FEVD at horizon H for returns (what % of return variance comes from trade shocks)
try:
    fevd = var.fevd(H)
    k = Y.shape[1]
    fevd_last = fevd.decomp[:, :, -1] * 100.0  # shape (k, k)
    idx_map = {n:i for i,n in enumerate(Y.columns)}
    rows = [idx_map['R_ETH'], idx_map['R_ARB'], idx_map['R_POLY']]
    cols = [idx_map['OF_ETH'], idx_map['OF_ARB'], idx_map['OF_POLY']]
    fevd_df = pd.DataFrame(fevd_last[np.ix_(rows, cols)],
                           index=['Var(R_ETH)','Var(R_ARB)','Var(R_POLY)'],
                           columns=['Shock(OF_ETH)','Shock(OF_ARB)','Shock(OF_POLY)']).round(2)
    print("\n=== FEVD @ 30 min (percent of return variance due to trade shocks) ===")
    print(fevd_df.to_string())

    # Leadership heuristic: each venue’s trade shock contribution to the OTHER venues’ returns
    contrib = {
        'ETH_trade': float(fevd_df.loc[['Var(R_ARB)','Var(R_POLY)'], 'Shock(OF_ETH)'].mean()),
        'ARB_trade': float(fevd_df.loc[['Var(R_ETH)','Var(R_POLY)'], 'Shock(OF_ARB)'].mean()),
        'POLY_trade': float(fevd_df.loc[['Var(R_ETH)','Var(R_ARB)'], 'Shock(OF_POLY)'].mean()),
    }
    print("\n=== Price discovery leadership (trade-shock contribution to others’ returns) ===")
    for k, v in contrib.items():
        print(f"{k}: {v:.2f}%")
    print("Leader (heuristic):", max(contrib, key=contrib.get))
except Exception as e:
    print("FEVD unavailable:", e)

# =========================
# 8) Add Liquidity-Provision (LP) shocks and re-estimate recursive SVAR (VAR+Cholesky)
# =========================

# --- 8.1 Helper: build signed LP shock per chain (mints +, burns -) at 1-min, USD z-score ---
def signed_lp_flow_1min(mints_df, burns_df, price_1m, label, lookback_z=60):
    """
    Heuristic LP proxy:
      • Use - for burns (liquidity leaves), + for mints (liquidity added).
      • Prefer ETH leg if present (amount1, delta_y); else use 'liquidity' column; else sums of amount0/amount1 magnitudes.
      • Convert to USD via minute price and z-score (rolling) for comparability.
    """
    def _parse_time(df):
        if df is None or df.empty: 
            return pd.Series(dtype='datetime64[ns]'), pd.Series(dtype=float)
        g = df.copy()
        g.columns = [c.lower() for c in g.columns]
        t = None
        for cand in ['timestamp','time','block_timestamp','blocktime','block_time','evt_block_time']:
            if cand in g.columns:
                t = pd.to_datetime(g[cand], utc=True, errors='coerce').dt.tz_localize(None)
                break
        if t is None:
            t = pd.to_datetime(pd.Series(range(len(g))), unit='s')
        # prefer ETH leg
        cand_val = None
        for eth_col in ['amount1','delta_y','amount1_in','amount1_out']:
            if eth_col in g.columns:
                cand_val = pd.to_numeric(g[eth_col], errors='coerce'); break
        if cand_val is None:
            if 'liquidity' in g.columns:
                cand_val = pd.to_numeric(g['liquidity'], errors='coerce')
            elif {'amount0','amount1'}.issubset(g.columns):
                a0 = pd.to_numeric(g['amount0'], errors='coerce').abs()
                a1 = pd.to_numeric(g['amount1'], errors='coerce').abs()
                cand_val = a0.combine(a1, lambda x,y: x+y)
            else:
                cand_val = pd.Series(index=g.index, dtype=float)
        return t, cand_val

    # + for mints, - for burns
    tm, vm = _parse_time(mints_df)
    tb, vb = _parse_time(burns_df)
    s_m = pd.Series(vm.values, index=tm).sort_index() if len(vm) else pd.Series(dtype=float)
    s_b = pd.Series(vb.values, index=tb).sort_index() if len(vb) else pd.Series(dtype=float)
    s = s_m.resample('min').sum(min_count=1).fillna(0.0) - s_b.resample('min').sum(min_count=1).fillna(0.0)

    # USD notional via on-chain minute price, then z-score
    px = price_1m.reindex(s.index).ffill()
    usd = s * px
    mu = usd.rolling(lookback_z, min_periods=max(10, lookback_z//5)).mean()
    sd = usd.rolling(lookback_z, min_periods=max(10, lookback_z//5)).std(ddof=0)
    z = (usd - mu) / sd
    z.name = f'LP_{label}'
    return z

# --- 8.2 Build LP series for each chain from your loaded CSVs ---
LP_ETH  = signed_lp_flow_1min(data.get('ethereum_mints'),  data.get('ethereum_burns'),  prices_1m['ETH_mainnet'], 'ETH')
LP_ARB  = signed_lp_flow_1min(data.get('arbitrum_mints'),  data.get('arbitrum_burns'),  prices_1m['ARB'],         'ARB')
LP_POLY = signed_lp_flow_1min(data.get('polygon_mints'),   data.get('polygon_burns'),   prices_1m['POLY'],        'POLY')

# --- 8.3 Ensure trades proxies (OF_*) exist from Step 7; if not, rebuild lightweight versions ---
def _ensure_of(name, swaps_df, px, label):
    if name in globals():
        return globals()[name]
    df = swaps_df.copy()
    df.columns = [c.lower() for c in df.columns]
    # time
    t = None
    for cand in ['timestamp','time','block_timestamp','blocktime','block_time','evt_block_time']:
        if cand in df.columns:
            t = pd.to_datetime(df[cand], utc=True, errors='coerce').dt.tz_localize(None)
            break
    if t is None:
        t = pd.to_datetime(pd.Series(range(len(df))), unit='s')
    # ETH leg (minus => buys positive)
    of_raw = None
    for eth_col in ['amount1','delta_y','amount1_in','amount1_out']:
        if eth_col in df.columns:
            of_raw = -pd.to_numeric(df[eth_col], errors='coerce'); break
    if of_raw is None and 'amount1' in df.columns:
        of_raw = -pd.to_numeric(df['amount1'], errors='coerce')
    if of_raw is None:
        return pd.Series(dtype=float, name=f'OF_{label}')
    s = pd.Series(of_raw.values, index=t).sort_index().resample('min').sum(min_count=1)
    px1m = px.reindex(s.index).ffill()
    of_usd = s * px1m
    mu = of_usd.rolling(60, min_periods=12).mean()
    sd = of_usd.rolling(60, min_periods=12).std(ddof=0)
    return ((of_usd - mu) / sd).rename(f'OF_{label}')

OF_ETH  = _ensure_of('OF_ETH',  data['ethereum_swaps'], prices_1m['ETH_mainnet'], 'ETH')
OF_ARB  = _ensure_of('OF_ARB',  data['arbitrum_swaps'], prices_1m['ARB'],         'ARB')
OF_POLY = _ensure_of('OF_POLY', data['polygon_swaps'],  prices_1m['POLY'],        'POLY')

# --- 8.4 Returns and alignment on strict 1-min grid ---
rets = np.log(common[['ETH_mainnet','ARB','POLY']]).diff()
rets.columns = ['R_ETH','R_ARB','R_POLY']

Y_lp = pd.concat([OF_ETH, OF_ARB, OF_POLY,  LP_ETH, LP_ARB, LP_POLY,  rets], axis=1).asfreq('min').dropna()
Y_lp.index = Y_lp.index.tz_localize(None)
Y_lp = Y_lp[['OF_ETH','OF_ARB','OF_POLY','LP_ETH','LP_ARB','LP_POLY','R_ETH','R_ARB','R_POLY']]

print("\n=== LP-augmented recursive SVAR input (head) ===")
print(Y_lp.head())
print("Samples:", len(Y_lp))

# For tractability with high dimension, cap sample if extremely long (keeps results real, not placeholders)
MAX_T = 200_000
if len(Y_lp) > MAX_T:
    Y_fit = Y_lp.tail(MAX_T)
    print(f"Using last {MAX_T:,} minutes for VAR fit to keep it tractable.")
else:
    Y_fit = Y_lp

# --- 8.X Sanitize & stabilize before VAR (avoid SVD non-convergence) ---
def _winsorize(df, q_low=0.001, q_high=0.999):
    lo = df.quantile(q_low, interpolation="nearest")
    hi = df.quantile(q_high, interpolation="nearest")
    return df.clip(lower=lo, upper=hi, axis=1)

def _zscore(df):
    mu = df.mean()
    sd = df.std(ddof=0)
    # avoid NaNs from zero-variance columns
    sd = sd.replace(0.0, np.nan)
    out = (df - mu) / sd
    return out

def _sanitize_for_var(df, winsor_low=0.001, winsor_high=0.999,
                      drop_tol=1e-10, max_nan_frac=0.2):
    Z = df.copy().replace([np.inf, -np.inf], np.nan)

    # 1) drop columns that are too sparse
    bad_nan = Z.columns[Z.isna().mean() > max_nan_frac]
    if len(bad_nan):
        print("Dropping high-NaN columns:", list(bad_nan))
        Z = Z.drop(columns=list(bad_nan))

    # 2) drop rows that still contain NaNs
    Z = Z.dropna(axis=0)
    if Z.empty:
        return Z

    # 3) drop (near-)constant columns
    const_cols = Z.columns[Z.std(ddof=0) < drop_tol]
    if len(const_cols):
        print("Dropping near-constant columns:", list(const_cols))
        Z = Z.drop(columns=list(const_cols))
    if Z.empty:
        return Z

    # 4) tame extremes and standardize
    Z = _winsorize(Z, winsor_low, winsor_high)
    Z = _zscore(Z)

    # 5) remove columns that still became all-NaN (e.g., zero variance)
    all_nan_cols = Z.columns[Z.isna().all()]
    if len(all_nan_cols):
        print("Dropping zero-variance columns (after standardization):", list(all_nan_cols))
        Z = Z.drop(columns=list(all_nan_cols))

    # 6) fill any remaining rare NaNs with 0 to keep rows
    Z = Z.fillna(0.0)
    return Z

# OPTIONAL: orthogonalize LP against same-chain OF to reduce collinearity
def _orthogonalize_lp_against_of(df):
    Z = df.copy()
    for ch in ['ETH','ARB','POLY']:
        of = Z.get(f'OF_{ch}')
        lp = Z.get(f'LP_{ch}')
        if of is None or lp is None:
            continue
        # simple projection: LP := LP - beta*OF (no intercept; data already z-scored later)
        beta = np.nan
        try:
            num = np.nanmean(of * lp)
            den = np.nanmean(of * of)
            beta = num / den if (den is not None and den != 0 and np.isfinite(den)) else 0.0
        except Exception:
            beta = 0.0
        Z[f'LP_{ch}'] = lp - beta * of
    return Z

print(f"Pre-clean rows: {len(Y_fit):,}, columns: {Y_fit.shape[1]}")

# (1) basic finite cast, optional LP~OF de-collinearity, then sanitize
Y_for_var = _orthogonalize_lp_against_of(Y_fit)
Y_for_var = _sanitize_for_var(Y_for_var)

print(f"Post-clean rows: {len(Y_for_var):,}, columns: {Y_for_var.shape[1]}")

# --- 8.Y Fit only if we still have data after cleaning ---
if Y_for_var.empty or len(Y_for_var) < 1000:
    print("Sanitized dataset too small after cleaning; skipping LP-augmented VAR on sanitized data.")
else:
    maxlags = min(4, max(2, len(Y_for_var)//200))
    aic = {}
    for p in range(1, maxlags+1):
        try:
            aic[p] = VAR(Y_for_var).fit(p, trend='c').aic
        except Exception:
            pass
    p_star = min(aic, key=aic.get) if aic else 2
    print(f"LP-augmented recursive SVAR (cleaned) chosen lag p = {p_star}")

    var_lp = VAR(Y_for_var).fit(p_star, trend='c')

    Sigma_u = var_lp.sigma_u
    P = np.linalg.cholesky(Sigma_u)
    labels = list(Y_for_var.columns)
    impact = pd.DataFrame(P, index=labels, columns=labels)
    print("\nContemporaneous impact (Cholesky, sanitized):")
    print(impact.round(4).to_string())

    H = 20
    irf_lp = var_lp.irf(H)

    def _print_irf_snapshot(shocks, title):
        print(f"\n=== IRF snapshot: Returns’ response to {title} shocks (horizon {H}) ===")
        for shock in shocks:
            if shock not in labels:
                continue
            j = labels.index(shock)
            print(f"\nShock: {shock}")
            for resp in ['R_ETH','R_ARB','R_POLY']:
                if resp not in labels:
                    continue
                i = labels.index(resp)
                path = irf_lp.irfs[:, i, j]
                t0 = float(path[0]); t1 = float(path[1])
                peak_idx = int(np.nanargmax(np.abs(path)))
                peak_val = float(path[peak_idx])
                print(f"{resp}: t=0={t0:+.4e}, t=1={t1:+.4e}, peak@t={peak_idx}={peak_val:+.4e}")

    _print_irf_snapshot([c for c in labels if c.startswith('OF_')], "Trade")
    _print_irf_snapshot([c for c in labels if c.startswith('LP_')], "LP")

    # --- FEVD on the sanitized system (normalize row-wise so shares sum to 100) ---
    try:
        fevd = var_lp.fevd(H)
        D = fevd.decomp[:, :, -1]  # shape (k, k); rows: variables, cols: shocks
        idx = {n:i for i,n in enumerate(labels)}

        if all(r in idx for r in ['R_ETH','R_ARB','R_POLY']):
            ret_rows   = np.array([idx['R_ETH'], idx['R_ARB'], idx['R_POLY']])
            trade_cols = np.array([i for n,i in idx.items() if n.startswith('OF_')])
            lp_cols    = np.array([i for n,i in idx.items() if n.startswith('LP_')])

            row_sum = D[ret_rows, :].sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = np.nan

            share_trade = (D[np.ix_(ret_rows, trade_cols)].sum(axis=1) / row_sum[:, 0]) * 100
            share_lp    = (D[np.ix_(ret_rows, lp_cols)].sum(axis=1)    / row_sum[:, 0]) * 100
            share_other = 100 - share_trade - share_lp

            fevd_df = pd.DataFrame({
                'Trade_shocks_%': np.clip(share_trade, 0, 100),
                'LP_shocks_%'   : np.clip(share_lp,    0, 100),
                'Other_%'       : np.clip(share_other, 0, 100)
            }, index=['Var(R_ETH)','Var(R_ARB)','Var(R_POLY)']).round(2)

            print("\n=== FEVD @ 30 min (Trade vs LP vs Other, %, sanitized, normalized) ===")
            print(fevd_df.to_string())
        else:
            print("\nFEVD skipped: one or more return series were dropped during cleaning.")
    except Exception as e:
        print("FEVD unavailable on sanitized data:", e)

    # === 9A. IRF error bands (95%) on the sanitized model ===
# === IRF confidence bands (version-safe & fast) ===
H_bands = 12   # shorter horizon for bands = big speedup; keep main IRFs at 20–30 if you like
REPL    = 20   # keep small to avoid long runs
SEED    = 42

def compute_irf_bands_fast(irf_obj, var_res, steps, repl, seed):
    """
    Try the quickest options first. Returns (lo, hi, method_str).
    Shapes: [steps, k, k]
    """
    # 1) Fast MC from VARResults; explicitly set T=steps and burn=0
    try:
        lo, hi = var_res.irf_errband_mc(
            orth=True, repl=repl, steps=steps, burn=0, seed=seed
        )
        return lo, hi, f"MC (repl={repl}, T={steps}, burn=0)"
    except KeyboardInterrupt:
        print("MC interrupted; trying SZ1 asymptotic bands…")
    except Exception as e:
        print("MC failed:", e)

    # 2) SZ1 bands from IRAnalysis (note underscore in method name)
    try:
        # Some versions accept repl/seed; others ignore them — both are fine.
        lo, hi = irf_obj.err_band_sz1(orth=True, signif=0.05, repl=repl, seed=seed)
        return lo, hi, "SZ1 (asymptotic)"
    except KeyboardInterrupt:
        print("SZ1 interrupted; trying IRAnalysis Monte Carlo…")
    except Exception as e:
        print("SZ1 failed:", e)

    # 3) IRAnalysis Monte Carlo (older signature)
    try:
        lo, hi = irf_obj.errband_mc(orth=True, repl=repl, T=steps, signif=0.05, seed=seed)
        return lo, hi, f"IRAnalysis MC (repl={repl}, T={steps})"
    except Exception as e:
        print("IRAnalysis MC failed:", e)

    return None, None, "none"

def _save_irf_bands(irf, lo, hi, labels, out_path):
    """
    Save IRFs and 95% bands. Handles mismatched horizons by using the minimum
    available across (irf, lo, hi).
    """
    H_eff = min(irf.irfs.shape[0], lo.shape[0], hi.shape[0])  # align horizons
    rows = []
    for j, shock in enumerate(labels):
        for i, resp in enumerate(labels):
            if not resp.startswith("R_"):
                continue
            for h in range(H_eff):
                rows.append({
                    "shock": shock, "response": resp, "h": h,
                    "irf": float(irf.irfs[h, i, j]),
                    "lo95": float(lo[h, i, j]),
                    "hi95": float(hi[h, i, j]),
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)

if 'var_lp' in locals() and 'irf_lp' in locals():
    lower95, upper95, band_method = compute_irf_bands_fast(irf_lp, var_lp, H_bands, REPL, SEED)
    print(f"IRF bands computed using: {band_method}")
    if (lower95 is not None) and (upper95 is not None):
        _save_irf_bands(irf_lp, lower95, upper95, labels, "sanitized_irf_bands.csv")
    else:
        print("Bands unavailable — skipping band CSV.")
else:
    print("Skipping IRF bands: sanitized VAR not estimated.")

# === 9B. Unscaled model on surviving columns (for interpretability) ===
cols_ok = list(Y_for_var.columns)
Y_fit_unscaled = Y_fit[cols_ok].dropna() if cols_ok else pd.DataFrame()

if cols_ok and len(Y_fit_unscaled) >= 1000:
    maxlags = min(4, max(2, len(Y_fit_unscaled)//200))
    aic_un = {}
    for p in range(1, maxlags+1):
        try:
            aic_un[p] = VAR(Y_fit_unscaled).fit(p, trend='c').aic
        except Exception:
            pass
    p_star_un = min(aic_un, key=aic_un.get) if aic_un else 2
    print(f"\nUnscaled VAR chosen lag p = {p_star_un}")

    var_lp_un = VAR(Y_fit_unscaled).fit(p_star_un, trend='c')
    Sigma_un = var_lp_un.sigma_u
    P_un = np.linalg.cholesky(Sigma_un)
    labels_un = list(Y_fit_unscaled.columns)

    impact_un = pd.DataFrame(P_un, index=labels_un, columns=labels_un)
    print("\nContemporaneous impact (Cholesky, UNscaled):")
    print(impact_un.round(4).to_string())

    H_un = 30
    irf_un = var_lp_un.irf(H_un)

    def _print_irf_snapshot_un(shock_list, title):
        print(f"\n=== IRF snapshot (UNscaled): Returns’ response to {title} shocks (h={H_un}) ===")
        for shock in shock_list:
            if shock not in labels_un:
                continue
            j = labels_un.index(shock)
            print(f"\nShock: {shock}")
            for resp in ['R_ETH','R_ARB','R_POLY']:
                if resp not in labels_un:
                    continue
                i = labels_un.index(resp)
                path = irf_un.irfs[:, i, j]
                t0, t1 = float(path[0]), float(path[1])
                pk = int(np.nanargmax(np.abs(path)))
                pv = float(path[pk])
                print(f"{resp}: t=0={t0:+.4e}, t=1={t1:+.4e}, peak@t={pk}={pv:+.4e}")

    _print_irf_snapshot_un([c for c in labels_un if c.startswith('OF_')], "Trade")
    _print_irf_snapshot_un([c for c in labels_un if c.startswith('LP_')], "LP")

    try:
        fevd_un = var_lp_un.fevd(H_un)
        D = fevd_un.decomp[:, :, -1]
        idx_un = {n:i for i,n in enumerate(labels_un)}
        if all(r in idx_un for r in ['R_ETH','R_ARB','R_POLY']):
            ret_rows   = np.array([idx_un['R_ETH'], idx_un['R_ARB'], idx_un['R_POLY']])
            trade_cols = np.array([i for n,i in idx_un.items() if n.startswith('OF_')])
            lp_cols    = np.array([i for n,i in idx_un.items() if n.startswith('LP_')])
            row_sum = D[ret_rows, :].sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = np.nan
            share_trade = (D[np.ix_(ret_rows, trade_cols)].sum(axis=1) / row_sum[:,0]) * 100
            share_lp    = (D[np.ix_(ret_rows, lp_cols)].sum(axis=1)    / row_sum[:,0]) * 100
            share_other = 100 - share_trade - share_lp
            fevd_df_un = pd.DataFrame({
                'Trade_shocks_%': np.clip(share_trade, 0, 100),
                'LP_shocks_%'   : np.clip(share_lp,    0, 100),
                'Other_%'       : np.clip(share_other, 0, 100)
            }, index=['Var(R_ETH)','Var(R_ARB)','Var(R_POLY)']).round(2)
            print("\n=== FEVD @ 30 min (UNscaled, normalized) ===")
            print(fevd_df_un.to_string())
        else:
            print("\nFEVD (UNscaled) skipped: some returns missing.")
    except Exception as e:
        print("FEVD (UNscaled) unavailable:", e)
else:
    print("9B skipped: not enough rows/columns after cleaning.")

# === Minimal ordering robustness on a smaller window ===
try:
    T_rob = 50_000
    Y_rob = Y_fit.tail(T_rob)

    alt_orderings = [
        ['OF_ARB','OF_ETH','OF_POLY','LP_ETH','LP_POLY','R_ETH','R_ARB','R_POLY'],  # swap ETH/ARB
        ['LP_ETH','OF_ETH','OF_ARB','OF_POLY','LP_POLY','R_ETH','R_ARB','R_POLY'],  # LP before OF
    ]

    for order in alt_orderings:
        cols = [c for c in order if c in Y_rob.columns]
        var_r = VAR(Y_rob[cols]).fit(min(4, max(2, len(Y_rob)//200)))
        fevd  = var_r.fevd(30).decomp[:, :, -1] * 100.0
        idx   = {n:i for i,n in enumerate(cols)}
        if all(k in idx for k in ['R_ETH','R_ARB','R_POLY','OF_ETH','OF_ARB','OF_POLY']):
            leader = {
                'ETH_trade': float(fevd[idx['R_ARB'], idx['OF_ETH']] + fevd[idx['R_POLY'], idx['OF_ETH']]) / 2.0,
                'ARB_trade': float(fevd[idx['R_ETH'], idx['OF_ARB']] + fevd[idx['R_POLY'], idx['OF_ARB']]) / 2.0,
                'POLY_trade': float(fevd[idx['R_ETH'], idx['OF_POLY']] + fevd[idx['R_ARB'], idx['OF_POLY']]) / 2.0,
            }
            print(f"\nOrdering check {cols}: leader = {max(leader, key=leader.get)} | {leader}")
except Exception as e:
    print("Ordering check skipped:", e)

if 'fevd_df' in locals():
    fevd_df.to_csv("fevd_sanitized_30m.csv")
if 'fevd_df_un' in locals():
    fevd_df_un.to_csv("fevd_unscaled_30m.csv")

# Stability (all roots inside unit circle)
# B) companion-matrix eigenvalues: < 1 means stable
def stability_report(var_res, tag):
    roots = np.asarray(var_res.roots)          # AR roots
    rho = float(np.max(np.abs(roots))) if roots.size else float("nan")
    print(f"{tag} VAR: max|root|={rho:.6f}  (stable if > 1) | is_stable={var_res.is_stable()}")

stability_report(var_lp, "Sanitized")
if 'var_lp_un' in locals():
    stability_report(var_lp_un, "UNscaled")

# Residual whiteness (Ljung–Box on each equation, a few lags)

#   - sanitized model:
res = pd.DataFrame(var_lp.resid, columns=Y_for_var.columns)
#   - or unscaled model:
# res = pd.DataFrame(var_lp_un.resid, columns=Y_fit_unscaled.columns)

lags = [12, 24]

# Run per column and assemble a tidy MultiIndex table
lb_tables = {}
for col in res.columns:
    out = acorr_ljungbox(res[col].dropna(), lags=lags, return_df=True)
    out.index = [f"lag{int(l)}" for l in out.index]   # prettier row names
    lb_tables[col] = out[["lb_stat", "lb_pvalue"]]

lb = pd.concat(lb_tables, axis=1)  # columns are a MultiIndex: (series, metric)

# quick, readable summary: p-values only
lb_pvals = lb.xs("lb_pvalue", axis=1, level=1)
print("\nLjung–Box p-values (rows = lag, cols = series)")
print(lb_pvals.to_string(float_format=lambda x: f"{x:0.3f}"))

# Heteroskedasticity & normality (report briefly)
# Use 'res' you already created from var_lp.resid
arch_p = res.apply(lambda s: het_arch(s.dropna(), nlags=12)[1])
norm_p = res.apply(lambda s: normal_ad(s.dropna())[1])

print("\nARCH test p-values (lag 12):")
print(arch_p.to_string(float_format=lambda x: f"{x:0.3f}"))
print("\nNormality (Anderson–Darling) p-values:")
print(norm_p.to_string(float_format=lambda x: f"{x:0.3f}"))

# (optional saves)
arch_p.to_csv("residual_arch_pvals.csv")
norm_p.to_csv("residual_normality_pvals.csv")

# 1) Load tidy bands file produced in 9A
bands_path = Path("sanitized_irf_bands.csv")
if not bands_path.exists():
    print("Bands file not found; skipping IRF plot.")
else:
    df = pd.read_csv(bands_path)
    # ... rest of plotting code unchanged ...

# Keep only returns as responses and OF shocks
resp_keep = ["R_ETH", "R_ARB", "R_POLY"]
shock_keep = ["OF_ETH", "OF_ARB", "OF_POLY"]
df = df[df["response"].isin(resp_keep) & df["shock"].isin(shock_keep)].copy()

# 2) Style (adjust once for the whole script)
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
})

# Colors/labels per shock
shock_color = {
    "OF_ETH": "#1f77b4",   # blue
    "OF_ARB": "#2ca02c",   # green
    "OF_POLY": "#d62728",  # red
}
shock_label = {
    "OF_ETH": "ETH OF shock",
    "OF_ARB": "ARB OF shock",
    "OF_POLY": "POLY OF shock",
}
resp_title = {
    "R_ETH": "Response: ETH returns",
    "R_ARB": "Response: ARB returns",
    "R_POLY": "Response: POLY returns",
}

# 3) Build figure: one panel per response, three lines (one per shock) + 95% band
responses = resp_keep
fig, axes = plt.subplots(len(responses), 1, figsize=(7.2, 8.6), sharex=True)

# Ensure axes is iterable even if len(responses)==1
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])

# Harmonize y-lims across panels (symmetric around 0)
ymax = 0.0
for r in responses:
    sub = df[df["response"] == r]
    if sub.empty:
        continue
    ymax = max(ymax, np.nanmax(np.abs(sub[["lo95", "hi95", "irf"]].values)))
if not np.isfinite(ymax) or ymax == 0:
    ymax = 1e-4  # fallback

# 4) Plot
for ax, r in zip(axes, responses):
    sub_r = df[df["response"] == r].copy()
    if sub_r.empty:
        ax.set_title(resp_title.get(r, r))
        ax.axhline(0, color="k", lw=1)
        continue

    # one line + band per shock
    for sh in shock_keep:
        sub_rs = sub_r[sub_r["shock"] == sh]
        if sub_rs.empty:
            continue
        h  = sub_rs["h"].values
        y  = sub_rs["irf"].values
        lo = sub_rs["lo95"].values
        hi = sub_rs["hi95"].values

        ax.plot(h, y, color=shock_color[sh], label=shock_label[sh])
        ax.fill_between(h, lo, hi, color=shock_color[sh], alpha=0.18, linewidth=0)

    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("IRF (return)")
    ax.set_title(resp_title.get(r, r))
    ax.set_ylim(-1.05*ymax, 1.05*ymax)

axes[-1].set_xlabel("Horizon (minutes)")
# Single shared legend at the top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.98])

# 5) Save
out_png = "irf_returns_to_OF_sanitized.png"
out_svg = "irf_returns_to_OF_sanitized.svg"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
print(f"Saved: {out_png} and {out_svg}")
plt.close(fig)

# =========================
# Dissertation Exports: Tables & Figures
# =========================
print("\n=== Creating dissertation tables & figures in ./artifacts ===")

# ---------- T1: Data coverage ----------
try:
    cov_rows = []
    for col in ['ETH_mainnet','ARB','POLY']:
        s = prices_1m[col].dropna()
        cov_rows.append({
            'Series': col,
            'First timestamp': s.index.min(),
            'Last timestamp':  s.index.max(),
            'Obs (minutes)':   int(s.shape[0]),
        })
    s_common = common.dropna(how='any')
    cov_rows.append({
        'Series': 'COMMON_INTERSECTION',
        'First timestamp': s_common.index.min(),
        'Last timestamp':  s_common.index.max(),
        'Obs (minutes)':   int(s_common.shape[0]),
    })
    T1 = pd.DataFrame(cov_rows)
    save_table(T1, "T1_data_coverage",
               "Sample coverage. Number of 1-minute observations by chain and common intersection period.",
               "tab:data_coverage", index=False)
except Exception as e:
    print("T1 error:", e)

# ---------- T2: Spread summary stats ----------
try:
    T2 = spreads[['ARB_vs_ETH_bps','POLY_vs_ETH_bps']].describe(percentiles=[.25,.5,.75]).rename(index={
        '25%':'p25','50%':'median','75%':'p75'
    })
    save_table(T2, "T2_spread_summary",
               "Cross-venue spreads vs Ethereum (bps): summary statistics.",
               "tab:spread_stats", index=True)
except Exception as e:
    print("T2 error:", e)

# ---------- T3: Arbitrage signal summary (overall + by market) ----------
try:
    # thresholds
    thr_df = pd.Series(thr, name="threshold_bps")
    # closure + minutes stats
    def _q(s,p): 
        try: return float(s.quantile(p))
        except: return np.nan
    def _mins_stats(s):
        s = s.dropna()
        return pd.Series({
            'p25': _q(s, .25), 'median': _q(s, .5), 'p75': _q(s, .75),
            'mean': float(s.mean()) if len(s) else np.nan,
            'min': float(s.min()) if len(s) else np.nan,
            'max': float(s.max()) if len(s) else np.nan
        })
    overall = pd.Series({
        'signals_total': int(convergence.shape[0]),
        'closed_count': int(convergence['closed'].sum()),
        'closed_rate_%': float(100*convergence['closed'].mean())
    })
    mins_overall = _mins_stats(convergence.loc[convergence['closed'],'minutes_to_close'])
    T3_overall = pd.concat([overall, mins_overall])
    # by market
    gb = convergence.groupby('market')
    T3_by_mkt = pd.DataFrame({
        'signals_total': gb.size(),
        'closed_count': gb['closed'].sum().astype(int),
        'closed_rate_%': gb['closed'].mean()*100
    })
    mins_by = gb.apply(lambda g: _mins_stats(g.loc[g['closed'],'minutes_to_close']))
    T3_by_mkt = T3_by_mkt.join(mins_by)
    # save
    save_table(thr_df.to_frame(), "T3a_thresholds",
               "Signal thresholds (bps) used to flag mispricings.", "tab:thresholds", index=True)
    save_table(T3_overall.to_frame(name='value'), "T3b_arb_overall",
               "Arbitrage signals and convergence outcomes (overall).", "tab:arb_overall", index=True)
    save_table(T3_by_mkt, "T3c_arb_by_market",
               "Arbitrage signals and convergence outcomes by market.", "tab:arb_by_market", index=True)
except Exception as e:
    print("T3 error:", e)

# ---------- F1: Time series of spreads (calm & stress 7-day windows) ----------
try:
    calm_start, calm_end, stress_start, stress_end = pick_windows_for_spreads(spreads)
    # Calm window
    if calm_start and calm_end:
        sub = spreads.loc[calm_start:calm_end]
        fig, ax = plt.subplots(figsize=(9,3.5))
        ax.plot(sub.index, sub['ARB_vs_ETH_bps'], label='ARB vs ETH')
        ax.plot(sub.index, sub['POLY_vs_ETH_bps'], label='POLY vs ETH')
        ax.axhline(thr['ARB_vs_ETH_bps'], ls='--'); ax.axhline(-thr['ARB_vs_ETH_bps'], ls='--')
        ax.axhline(thr['POLY_vs_ETH_bps'], ls='--'); ax.axhline(-thr['POLY_vs_ETH_bps'], ls='--')
        ax.set_title("Spreads (bps) — calm window")
        ax.set_ylabel("bps"); ax.grid(True, alpha=.25); ax.legend()
        save_figure(fig, "F1_spreads_calm",
                    "Minute-by-minute spreads (bps) over a calm 7-day window; dashed lines mark signal thresholds.",
                    "fig:spreads_calm")
        plt.close(fig)
    # Stress window
    if stress_start and stress_end:
        sub = spreads.loc[stress_start:stress_end]
        fig, ax = plt.subplots(figsize=(9,3.5))
        ax.plot(sub.index, sub['ARB_vs_ETH_bps'], label='ARB vs ETH')
        ax.plot(sub.index, sub['POLY_vs_ETH_bps'], label='POLY vs ETH')
        ax.axhline(thr['ARB_vs_ETH_bps'], ls='--'); ax.axhline(-thr['ARB_vs_ETH_bps'], ls='--')
        ax.axhline(thr['POLY_vs_ETH_bps'], ls='--'); ax.axhline(-thr['POLY_vs_ETH_bps'], ls='--')
        ax.set_title("Spreads (bps) — stress window")
        ax.set_ylabel("bps"); ax.grid(True, alpha=.25); ax.legend()
        save_figure(fig, "F1_spreads_stress",
                    "Minute-by-minute spreads (bps) over a stress 7-day window; dashed lines mark signal thresholds.",
                    "fig:spreads_stress")
        plt.close(fig)
except Exception as e:
    print("F1 error:", e)

# ---------- F2: ECDF of minutes-to-close by market ----------
try:
    closed = convergence[convergence['closed']].copy()
    fig, ax = plt.subplots(figsize=(6,4))
    for mkt, g in closed.groupby('market'):
        x, F = ecdf(g['minutes_to_close'].astype(float))
        if x.size: ax.plot(x, F, label=mkt.title())
    ax.set_xlabel("Minutes to close"); ax.set_ylabel("ECDF"); ax.grid(True, alpha=.25); ax.legend()
    save_figure(fig, "F2_ecdf_minutes_to_close",
                "Convergence speed: ECDF of minutes-to-close by market.", "fig:ecdf_close")
    plt.close(fig)
except Exception as e:
    print("F2 error:", e)

# ---------- T4: Largest initial mispricings (Top 10) ----------
try:
    biggest = convergence.reindex(convergence['spread0_bp'].abs().sort_values(ascending=False).index).head(10)
    cols = ['ts','market','spread0_bp','band_bp','minutes_to_close','closed','peak_abs_bp','auc_bpmin']
    save_table(biggest[cols], "T4_top_mispricings",
               "Largest initial mispricings (absolute bps), with convergence outcomes.", "tab:top_mispricings", index=False)
except Exception as e:
    print("T4 error:", e)

# ---------- T5: VAR lag selection (AIC) ----------
try:
    # Base (trades-first) on Y (if present)
    if 'Y' in globals() and isinstance(Y, pd.DataFrame) and len(Y) > 2000:
        maxlags = min(6, max(2, len(Y)//200))
        aic_base = {}
        for p in range(1, maxlags+1):
            try: aic_base[p] = VAR(Y).fit(p).aic
            except: pass
        T5A = pd.Series(aic_base, name='AIC').to_frame()
        save_table(T5A, "T5a_aic_base",
                   "Reduced-form VAR lag selection (base trades-first model) by AIC.", "tab:aic_base", index=True)
    # LP-augmented (sanitized) on Y_for_var (if present)
    if 'Y_for_var' in globals() and isinstance(Y_for_var, pd.DataFrame) and len(Y_for_var) > 2000:
        maxlags = min(4, max(2, len(Y_for_var)//200))
        aic_lp = {}
        for p in range(1, maxlags+1):
            try: aic_lp[p] = VAR(Y_for_var).fit(p, trend='c').aic
            except: pass
        T5B = pd.Series(aic_lp, name='AIC').to_frame()
        save_table(T5B, "T5b_aic_lp",
                   "Reduced-form VAR lag selection (LP-augmented sanitized model) by AIC.", "tab:aic_lp", index=True)
except Exception as e:
    print("T5 error:", e)

# ---------- T6: Contemporaneous impact matrix (sanitized) ----------
try:
    if 'impact' in globals() and isinstance(impact, pd.DataFrame):
        save_table(impact.round(4), "T6_impact_sanitized",
                   "Cholesky contemporaneous impact matrix (sanitized LP-augmented model).",
                   "tab:impact_sanitized", index=True)
except Exception as e:
    print("T6 error:", e)

# ---------- F3: IRFs (returns to trade shocks) with bands ----------
try:
    # Use the figure you already saved earlier; copy into artifacts/ and write LaTeX snippet
    src = Path("irf_returns_to_OF_sanitized.png")
    if src.exists():
        dst = OUT / src.name
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        save_caption_snippet_for_image(dst.name, "F3_irf_returns_to_OF",
            "Orthogonalized IRFs (95\\% Monte Carlo bands): returns’ response to trade-flow shocks; horizon 20–30 minutes.",
            "fig:irf_trade")
except Exception as e:
    print("F3 error:", e)

# ---------- T7: FEVD @30m (trade shocks only, base) ----------
try:
    if 'var' in globals():
        fevd_base = var.fevd(30)
        k = var.neqs
        idx_map = {n:i for i,n in enumerate(var.names)}
        rows = [idx_map.get('R_ETH'), idx_map.get('R_ARB'), idx_map.get('R_POLY')]
        cols = [idx_map.get('OF_ETH'), idx_map.get('OF_ARB'), idx_map.get('OF_POLY')]
        rows = [r for r in rows if r is not None]; cols = [c for c in cols if c is not None]
        if rows and cols:
            D = fevd_base.decomp[:, :, -1] * 100.0
            T7 = pd.DataFrame(D[np.ix_(rows, cols)], 
                              index=[var.names[r] for r in rows],
                              columns=[f"Shock({var.names[c]})" for c in cols]).round(2)
            save_table(T7, "T7_fevd_base_trade",
                       "FEVD at 30 minutes (base model): share of return variance explained by trade shocks (%).",
                       "tab:fevd_base", index=True)
except Exception as e:
    print("T7 error:", e)

# ---------- T8: FEVD @30m (Trade vs LP vs Other, sanitized) ----------
try:
    if 'var_lp' in globals() and 'Y_for_var' in globals():
        fevd = var_lp.fevd(30).decomp[:, :, -1]
        labels = list(Y_for_var.columns)
        idx = {n:i for i,n in enumerate(labels)}
        if all(k in idx for k in ['R_ETH','R_ARB','R_POLY']):
            ret_rows   = np.array([idx['R_ETH'], idx['R_ARB'], idx['R_POLY']])
            trade_cols = np.array([i for n,i in idx.items() if n.startswith('OF_')])
            lp_cols    = np.array([i for n,i in idx.items() if n.startswith('LP_')])
            row_sum = fevd[ret_rows, :].sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = np.nan
            share_trade = (fevd[np.ix_(ret_rows, trade_cols)].sum(axis=1) / row_sum[:,0]) * 100
            share_lp    = (fevd[np.ix_(ret_rows, lp_cols)].sum(axis=1)    / row_sum[:,0]) * 100
            share_other = 100 - share_trade - share_lp
            T8 = pd.DataFrame({
                'Trade_shocks_%': np.clip(share_trade, 0, 100),
                'LP_shocks_%'   : np.clip(share_lp,    0, 100),
                'Other_%'       : np.clip(share_other, 0, 100)
            }, index=['Var(R_ETH)','Var(R_ARB)','Var(R_POLY)']).round(2)
            save_table(T8, "T8_fevd_lp_trade_lp_other",
                       "FEVD at 30 minutes (LP-augmented sanitized model): trade vs LP vs other (%).",
                       "tab:fevd_lp", index=True)
except Exception as e:
    print("T8 error:", e)

# =========================
# Appendix (Robustness)
# =========================

# ---------- A1: Full convergence stats by market ----------
try:
    def _stats_block(g):
        s = g[g['closed']]['minutes_to_close'].astype(float)
        return pd.Series({
            'signals_total': int(g.shape[0]),
            'closed_count': int(g['closed'].sum()),
            'closed_rate_%': 100*float(g['closed'].mean()),
            'p25': float(s.quantile(.25)) if len(s) else np.nan,
            'median': float(s.quantile(.5)) if len(s) else np.nan,
            'p75': float(s.quantile(.75)) if len(s) else np.nan,
            'mean': float(s.mean()) if len(s) else np.nan,
            'min': float(s.min()) if len(s) else np.nan,
            'max': float(s.max()) if len(s) else np.nan,
            'avg_peak_abs_bp': float(g['peak_abs_bp'].mean()),
            'avg_auc_bpmin': float(g['auc_bpmin'].mean()),
        })
    A1 = convergence.groupby('market').apply(_stats_block)
    save_table(A1, "A1_convergence_by_market",
               "Convergence statistics by market (counts, closure rates, minutes-to-close, peak |bps|, and bp-min area).",
               "tab:conv_by_mkt", index=True)
except Exception as e:
    print("A1 error:", e)

# ---------- A2: Slowest closures (Top 50) ----------
try:
    tt = convergence.copy()
    tt['_tt_sort'] = np.where(tt['closed'], tt['minutes_to_close'], np.inf)
    slowest = tt.sort_values(['_tt_sort','ts','market']).head(50).drop(columns=['_tt_sort'])
    save_table(slowest[['ts','market','spread0_bp','band_bp','minutes_to_close','closed','peak_abs_bp','auc_bpmin']],
               "A2_slowest_closures_top50",
               "Slowest closures (top 50 by minutes to close; non-closed appear at bottom as ∞).",
               "tab:slowest_50", index=False)
except Exception as e:
    print("A2 error:", e)

# ---------- A3: Ordering robustness table ----------
try:
    rob_rows = []
    # Repeat the two checks you already ran
    alt_orderings = [
        ['OF_ARB','OF_ETH','OF_POLY','LP_ETH','LP_POLY','R_ETH','R_ARB','R_POLY'],
        ['LP_ETH','OF_ETH','OF_ARB','OF_POLY','LP_POLY','R_ETH','R_ARB','R_POLY'],
    ]
    if 'Y_fit' in globals():
        Y_rob = Y_fit.tail(50_000) if len(Y_fit) > 50_000 else Y_fit
        for order in alt_orderings:
            cols = [c for c in order if c in Y_rob.columns]
            var_r = VAR(Y_rob[cols]).fit(min(4, max(2, len(Y_rob)//200)))
            D = var_r.fevd(30).decomp[:, :, -1] * 100.0
            idx   = {n:i for i,n in enumerate(cols)}
            def get(c): return idx.get(c, None)
            if all(get(k) is not None for k in ['R_ETH','R_ARB','R_POLY','OF_ETH','OF_ARB','OF_POLY']):
                ETH_trade = float(D[get('R_ARB'), get('OF_ETH')] + D[get('R_POLY'), get('OF_ETH')]) / 2.0
                ARB_trade = float(D[get('R_ETH'), get('OF_ARB')] + D[get('R_POLY'), get('OF_ARB')]) / 2.0
                POLY_trade = float(D[get('R_ETH'), get('OF_POLY')] + D[get('R_ARB'), get('OF_POLY')]) / 2.0
                leader = max({'ETH_trade':ETH_trade,'ARB_trade':ARB_trade,'POLY_trade':POLY_trade}, key=lambda k: {'ETH_trade':ETH_trade,'ARB_trade':ARB_trade,'POLY_trade':POLY_trade}[k])
                rob_rows.append({'ordering': ' -> '.join(cols),
                 'ETH_trade_%': ETH_trade,
                 'ARB_trade_%': ARB_trade,
                 'POLY_trade_%': POLY_trade,
                 'leader': leader})

    if rob_rows:
        A3 = pd.DataFrame(rob_rows)
        save_table(A3, "A3_ordering_robustness",
                   "Ordering robustness: trade-shock contribution to other venues’ returns under alternative recursive orderings.",
                   "tab:ordering_robustness", index=False)
except Exception as e:
    print("A3 error:", e)

# ---------- A4: Model diagnostics ----------
try:
    # Sanitize residuals: use sanitized model by default
    res = pd.DataFrame(var_lp.resid, columns=Y_for_var.columns) if 'var_lp' in globals() else None
    # Ljung–Box p-values
    if res is not None:
        lags = [12, 24]
        lb_tables = {}
        for col in res.columns:
            out = acorr_ljungbox(res[col].dropna(), lags=lags, return_df=True)
            out.index = [f"lag{int(l)}" for l in out.index]
            lb_tables[col] = out[["lb_stat", "lb_pvalue"]]
        lb = pd.concat(lb_tables, axis=1)
        lb_p = lb.xs("lb_pvalue", axis=1, level=1)
        save_table(lb_p, "A4_ljungbox_pvals",
                   "Residual whiteness (Ljung–Box) p-values at lags 12 and 24.", "tab:lb_pvals", index=True)
        # ARCH test p-values (lag 12)
        arch_p = res.apply(lambda s: het_arch(s.dropna(), nlags=12)[1])
        save_table(arch_p.to_frame("p_value"), "A4_arch_pvalues",
                   "ARCH heteroskedasticity test p-values (lag 12).", "tab:arch_pvals", index=True)
        # Normality (D’Agostino K^2) p-values
        norm_p = res.apply(lambda s: normaltest(s.dropna()).pvalue)
        save_table(norm_p.to_frame("p_value"), "A4_normality_pvalues",
                   "Normality test (D’Agostino K^2) p-values.", "tab:normality_pvals", index=True)
    # Stability flags (booleans; roots magnitudes vary across versions)
    stab_rows = []
    if 'var_lp' in globals():
        try: stab_rows.append({'model':'Sanitized (LP-augmented)', 'is_stable': bool(var_lp.is_stable())})
        except: stab_rows.append({'model':'Sanitized (LP-augmented)', 'is_stable': np.nan})
    if 'var_lp_un' in globals():
        try: stab_rows.append({'model':'UNscaled', 'is_stable': bool(var_lp_un.is_stable())})
        except: stab_rows.append({'model':'UNscaled', 'is_stable': np.nan})
    if stab_rows:
        save_table(pd.DataFrame(stab_rows), "A4_stability_flags",
                   "VAR stability flags (all roots inside unit circle).", "tab:stability_flags", index=False)
except Exception as e:
    print("A4 error:", e)

# ---------- A1 Figure: Histograms of spreads ----------
try:
    fig, axes = plt.subplots(2, 1, figsize=(7,6), sharex=True)
    axes[0].hist(spreads['ARB_vs_ETH_bps'].dropna(), bins=100)
    axes[0].set_title("ARB vs ETH spreads (bps)"); axes[0].grid(True, alpha=.25)
    axes[1].hist(spreads['POLY_vs_ETH_bps'].dropna(), bins=100)
    axes[1].set_title("POLY vs ETH spreads (bps)"); axes[1].grid(True, alpha=.25)
    axes[1].set_xlabel("bps")
    save_figure(fig, "A1_spread_histograms",
                "Distribution of spreads (bps) per venue.", "fig:spread_hist")
    plt.close(fig)
except Exception as e:
    print("A1 figure error:", e)

# ---------- A2 Figure: Minutes-to-close distribution ----------
try:
    closed = convergence[convergence['closed']]
    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
    for ax, mkt in zip(axes, ['arbitrum','polygon']):
        s = closed.loc[closed['market']==mkt, 'minutes_to_close'].astype(float)
        ax.boxplot(s.values, vert=True, showfliers=False)
        ax.set_title(f"{mkt.title()}"); ax.set_ylabel("Minutes"); ax.grid(True, alpha=.25)
    save_figure(fig, "A2_minutes_to_close_box",
                "Minutes-to-close distribution by market (boxplots, outliers suppressed).",
                "fig:mtc_box")
    plt.close(fig)
except Exception as e:
    print("A2 figure error:", e)

# ---------- A3 Figure: Heatmap of impact matrix (sanitized) ----------
try:
    if 'impact' in globals():
        M = impact.values.astype(float)
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(M, aspect='auto')
        ax.set_xticks(range(impact.shape[1])); ax.set_xticklabels(impact.columns, rotation=45, ha='right')
        ax.set_yticks(range(impact.shape[0])); ax.set_yticklabels(impact.index)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha='center', va='center', fontsize=8)
        ax.set_title("Contemporaneous impact (Cholesky, sanitized)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_figure(fig, "A3_impact_heatmap",
                    "Heatmap of the contemporaneous impact matrix (sanitized LP-augmented model).",
                    "fig:impact_heatmap")
        plt.close(fig)
except Exception as e:
    print("A3 figure error:", e)

print("All artifacts written to:", OUT.resolve())
