"""
Exploratory Data Analysis
Generates all figures and tables for Section IV of the report:
panel structure, return distributions, serial dependence, volatility scaling,
and stationarity testing.
Requires: preprocessed_data.pkl, cohort CSVs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

OUT = Path('eda_outputs')
OUT.mkdir(exist_ok=True)
FIG_DPI = 300
SEED = 42
np.random.seed(SEED)

TIER_COLORS = {'Mega-Cap': '#2c7bb6', 'Large-Cap': '#fdae61', 'Mid-Cap': '#d7191c'}
TIER_ORDER = ['Mega-Cap', 'Large-Cap', 'Mid-Cap']

print("Loading data...")
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

log_ret_wide = data['log_returns']
scaled_wide = data['scaled_returns']
ewma_vol = data['ewma_vol']
permnos = data['permnos']
splits = data['splits']

c90 = pd.read_csv('cohort_1990_stocks.csv')
c24 = pd.read_csv('cohort_2024_stocks.csv')
meta = pd.concat([
    c90[['PERMNO', 'Ticker', 'Company Name', 'Tier']].assign(Cohort='1990'),
    c24[['PERMNO', 'Ticker', 'Company Name', 'Tier']].assign(Cohort='2024')
]).drop_duplicates(subset='PERMNO', keep='first')
meta = meta.set_index('PERMNO')

permno_to_tier = meta['Tier'].to_dict()
permno_to_ticker = meta['Ticker'].to_dict()


def wide_to_long(wide_df, value_name):
    long = wide_df.stack().reset_index()
    long.columns = ['Date', 'PERMNO', value_name]
    long['PERMNO'] = long['PERMNO'].astype(int)
    long['Ticker'] = long['PERMNO'].map(permno_to_ticker)
    long['Cap_Tier'] = long['PERMNO'].map(permno_to_tier)
    return long


df_raw = wide_to_long(log_ret_wide, 'log_return')
df_sc = wide_to_long(scaled_wide, 'scaled_return')
df_vol = wide_to_long(ewma_vol, 'trailing_vol')

df = df_raw.merge(df_sc[['Date', 'PERMNO', 'scaled_return']],
                  on=['Date', 'PERMNO'], how='left')
df = df.merge(df_vol[['Date', 'PERMNO', 'trailing_vol']],
              on=['Date', 'PERMNO'], how='left')
df = df.dropna(subset=['log_return'])

unmatched = df[df['Cap_Tier'].isna()]['PERMNO'].unique()
if len(unmatched) > 0:
    print(f"  WARNING: {len(unmatched)} PERMNOs missing Cap_Tier")
    df['Cap_Tier'] = df['Cap_Tier'].fillna('Unknown')

print(f"Loaded {df['PERMNO'].nunique()} assets, {len(df):,} rows")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")


# ── 4.1 Panel Structure ──────────────────────────────────────────────

print("=" * 60)
print("4.1 Panel Structure and Data Availability")
print("=" * 60)

active = (df.groupby('Date')['PERMNO'].nunique()
          .rename('active_stocks').reset_index())

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(active['Date'], active['active_stocks'],
                alpha=0.35, color='steelblue')
ax.plot(active['Date'], active['active_stocks'], linewidth=0.7, color='steelblue')
ax.set_ylabel('Number of Active Stocks')
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(0, df['PERMNO'].nunique() + 10)
ax.axhline(df['PERMNO'].nunique(), ls='--', c='grey', lw=0.7,
           label=f'Full universe ({df["PERMNO"].nunique()})')

for start, end, lbl, clr in [
    (splits['train'][0], splits['train'][1], 'Train', '#d4edda'),
    (splits['val'][0],   splits['val'][1],   'Val',   '#fff3cd'),
    (splits['test'][0],  splits['test'][1],  'Test',  '#f8d7da'),
]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               alpha=0.15, color=clr, label=lbl)

ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'fig1_active_stocks.png', dpi=FIG_DPI)
plt.close()
print("  -> fig1_active_stocks.png")


# ── 4.2 Return Distribution Diagnostics ──────────────────────────────

print("=" * 60)
print("4.2 Return Distribution Diagnostics")
print("=" * 60)


def return_stats(s):
    s = s.dropna()
    jb_stat, jb_p = stats.jarque_bera(s)
    return pd.Series({
        'N': len(s),
        'Mean (bps/day)': s.mean() * 1e4,
        'Std Dev (%)': s.std() * 100,
        'Skewness': stats.skew(s),
        'Excess Kurtosis': stats.kurtosis(s),
        'Min (%)': s.min() * 100,
        'Max (%)': s.max() * 100,
        'JB Stat': jb_stat,
        'JB p-value': jb_p,
    })


tier_rows = []
for tier in TIER_ORDER:
    sub = df[df['Cap_Tier'] == tier]['log_return']
    row = return_stats(sub)
    row['Cap_Tier'] = tier
    tier_rows.append(row)
full_row = return_stats(df['log_return'])
full_row['Cap_Tier'] = 'All'
tier_rows.append(full_row)
tier_table = pd.DataFrame(tier_rows).set_index('Cap_Tier')
tier_table.to_csv(OUT / 'table2_return_distributions.csv')
print("  -> table2_return_distributions.csv")

# QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
sample = df['log_return'].dropna().values
if len(sample) > 500_000:
    rng = np.random.default_rng(SEED)
    sample = rng.choice(sample, 500_000, replace=False)
theoretical_q = np.linspace(0.001, 0.999, 1000)
empirical_quantiles = np.quantile(sample, theoretical_q)
normal_quantiles = stats.norm.ppf(theoretical_q,
                                  loc=sample.mean(), scale=sample.std())
ax.scatter(normal_quantiles, empirical_quantiles, s=3, alpha=0.5,
           color='steelblue')
lims = [min(normal_quantiles.min(), empirical_quantiles.min()),
        max(normal_quantiles.max(), empirical_quantiles.max())]
ax.plot(lims, lims, '--', c='red', lw=1, label='Normal reference')
ax.set_xlabel('Theoretical Normal Quantiles')
ax.set_ylabel('Empirical Quantiles')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'fig4_qq_plot.png', dpi=FIG_DPI)
plt.close()
print("  -> fig4_qq_plot.png")


# ── 4.3 Serial Dependence ────────────────────────────────────────────

print("=" * 60)
print("4.3 Evidence of Serial Dependence")
print("=" * 60)

MAX_LAGS = 60


def aggregate_returns(group, freq_days):
    vals = group['log_return'].dropna().values
    n = len(vals) - (len(vals) % freq_days)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(vals[:n].reshape(-1, freq_days).sum(axis=1))


# ACF of absolute returns (volatility clustering)
df['abs_return'] = df['log_return'].abs()
ac_abs, ci_abs = acf(df['abs_return'].dropna().values,
                     nlags=MAX_LAGS, alpha=0.05)

fig, ax = plt.subplots(figsize=(10, 4.5))
lags = np.arange(1, MAX_LAGS + 1)
ax.bar(lags, ac_abs[1:], width=0.6, color='#d7191c', alpha=0.7)
se = 1.96 / np.sqrt(len(df['abs_return'].dropna()))
ax.axhline(se, ls='--', c='blue', lw=0.8, alpha=0.6, label='95% CI')
ax.axhline(-se, ls='--', c='blue', lw=0.8, alpha=0.6)
ax.axhline(0, c='black', lw=0.5)
ax.set_xlabel('Lag (trading days)')
ax.set_ylabel('Autocorrelation')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'fig6_acf_abs_returns.png', dpi=FIG_DPI)
plt.close()
print("  -> fig6_acf_abs_returns.png")

# Multi-horizon autocorrelation table
horizons = [1, 5, 10, 21, 42, 63]
horizon_labels = {1: '1d', 5: '1w', 10: '2w', 21: '1m', 42: '2m', 63: '3m'}
horizon_acf = []
for h in horizons:
    if h == 1:
        per_asset = (df.groupby('PERMNO')['log_return']
                     .apply(lambda s: s.dropna().autocorr(lag=1)))
    else:
        def acf1_at_horizon(grp, freq=h):
            agg = aggregate_returns(grp, freq)
            if len(agg) < 30:
                return np.nan
            return agg.autocorr(lag=1)
        per_asset = df.groupby('PERMNO').apply(acf1_at_horizon)

    horizon_acf.append({
        'Horizon (days)': h,
        'Horizon (approx)': horizon_labels[h],
        'Mean ACF(1)': per_asset.mean(),
        'Median ACF(1)': per_asset.median(),
        'Pct Positive': (per_asset > 0).mean() * 100,
        'N Assets': per_asset.notna().sum(),
    })

horizon_table = pd.DataFrame(horizon_acf).round(4)
horizon_table.to_csv(OUT / 'table4_horizon_acf.csv', index=False)
print("  -> table4_horizon_acf.csv")


# ── 4.4 Volatility Scaling Impact ────────────────────────────────────

print("=" * 60)
print("4.4 Impact of Volatility Scaling")
print("=" * 60)

cs_raw = df.groupby('Date')['log_return'].std().rename('raw')
cs_scaled = df.groupby('Date')['scaled_return'].std().rename('scaled')
cs = pd.concat([cs_raw, cs_scaled], axis=1).dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(cs.index, cs['raw'], lw=0.4, color='steelblue')
axes[0].set_ylabel('Cross-Sectional Std')
axes[0].set_title('(a) Raw Log Returns')
axes[0].axhline(cs['raw'].mean(), ls='--', c='grey', lw=0.7,
                label=f'Mean = {cs["raw"].mean():.4f}')
axes[0].legend(fontsize=8)

axes[1].plot(cs.index, cs['scaled'], lw=0.4, color='#d7191c')
axes[1].set_ylabel('Cross-Sectional Std')
axes[1].set_title('(b) Volatility-Scaled Returns')
axes[1].axhline(cs['scaled'].mean(), ls='--', c='grey', lw=0.7,
                label=f'Mean = {cs["scaled"].mean():.4f}')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'fig8_variance_homogenization.png', dpi=FIG_DPI,
            bbox_inches='tight')
plt.close()
print("  -> fig8_variance_homogenization.png")

# ACF preservation scatter
acf1_raw = (df.groupby(['PERMNO', 'Cap_Tier'])['log_return']
            .apply(lambda s: s.dropna().autocorr(lag=1))
            .reset_index().rename(columns={'log_return': 'ACF1_raw'}))
acf1_scaled = (df.groupby(['PERMNO', 'Cap_Tier'])['scaled_return']
               .apply(lambda s: s.dropna().autocorr(lag=1))
               .reset_index().rename(columns={'scaled_return': 'ACF1_scaled'}))
acf_compare = acf1_raw.merge(acf1_scaled, on=['PERMNO', 'Cap_Tier'])

fig, ax = plt.subplots(figsize=(6, 6))
for tier in TIER_ORDER:
    sub = acf_compare[acf_compare['Cap_Tier'] == tier]
    ax.scatter(sub['ACF1_raw'], sub['ACF1_scaled'], alpha=0.6,
               c=TIER_COLORS[tier], edgecolor='white', s=50, label=tier)
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, '--', c='grey', lw=1, label='45° line')
ax.set_xlabel('ACF(1) — Raw Log Returns')
ax.set_ylabel('ACF(1) — Volatility-Scaled Returns')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'fig11_acf_preservation.png', dpi=FIG_DPI)
plt.close()
print("  -> fig11_acf_preservation.png")

corr_preserve = acf_compare[['ACF1_raw', 'ACF1_scaled']].corr().iloc[0, 1]
print(f"  ACF(1) correlation (raw vs scaled): {corr_preserve:.4f}")


# ── 4.5 Stationarity Testing ─────────────────────────────────────────

print("=" * 60)
print("4.5 Stationarity Testing (ADF)")
print("=" * 60)

adf_results = []
for (permno, tier), grp in df.groupby(['PERMNO', 'Cap_Tier']):
    for col, label in [('log_return', 'Raw'), ('scaled_return', 'Scaled')]:
        series = grp[col].dropna()
        if len(series) < 100:
            continue
        try:
            stat, pval, lags_used, nobs, *_ = adfuller(
                series, maxlag=21, autolag='AIC')
            adf_results.append({
                'PERMNO': permno, 'Cap_Tier': tier, 'Series': label,
                'ADF Stat': round(stat, 4), 'p-value': round(pval, 6),
                'Reject 5%': pval < 0.05,
            })
        except Exception as e:
            print(f"  ADF failed for {permno} ({label}): {e}")

adf_df = pd.DataFrame(adf_results)
adf_overall = (adf_df.groupby('Series')['Reject 5%']
               .agg(n_assets='count', n_reject='sum', pct_reject='mean').round(3))
adf_overall['pct_reject'] = (adf_overall['pct_reject'] * 100).round(1)
adf_overall.to_csv(OUT / 'table6_adf_overall.csv')
print("  -> table6_adf_overall.csv")
print(adf_overall)

print("\n" + "=" * 60)
print("ALL EDA OUTPUTS SAVED TO ./eda_outputs/")
print("=" * 60)
