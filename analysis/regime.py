"""
Regime-Conditional Performance Analysis
Evaluates L252_h21 model performance across distinct market regimes
(pre-COVID, crash, recovery, rate hiking, etc.) and generates rolling
IC / hit-rate figures.
Requires: preprocessed_data.pkl, trained L252_h21 model
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path('models_v2')
OUT = Path('results_v2')
OUT.mkdir(exist_ok=True)
FIG_DPI = 300

# ── Data Loading ──────────────────────────────────────────────────────

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

scaled_wide = data['scaled_returns']
log_ret_wide = data['log_returns']
ewma_vol = data['ewma_vol']
splits = data['splits']
dates = scaled_wide.index
n_assets = scaled_wide.shape[1]

scaled_vals = scaled_wide.to_numpy(dtype=np.float64, na_value=np.nan)
log_ret_vals = log_ret_wide.to_numpy(dtype=np.float64, na_value=np.nan)
vol_vals = ewma_vol.to_numpy(dtype=np.float64, na_value=np.nan)
scaled_filled = np.nan_to_num(scaled_vals, nan=0.0)


def date_to_idx(d):
    return min(dates.searchsorted(pd.Timestamp(d)), len(dates) - 1)


# ── Model Architecture ────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, L, patch_len, stride, d_model):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = int((L - patch_len) / stride) + 1
        self.pad_len = max(0, (self.n_patches - 1) * stride + patch_len - L)
        self.padding = nn.ReplicationPad1d((0, self.pad_len))
        self.n_patches = int((L + self.pad_len - patch_len) / stride) + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B * C, 1, L)
        x = self.padding(x).squeeze(1)
        x = x.unfold(1, self.patch_len, self.stride)
        return self.proj(x) + self.pos_enc, B, C


class PatchTSTv2(nn.Module):
    def __init__(self, L, patch_len, stride, d_model, n_heads,
                 n_layers, d_ff, dropout, n_assets):
        super().__init__()
        self.patch_embed = PatchEmbedding(L, patch_len, stride, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L = x.shape
        z, _, _ = self.patch_embed(x)
        z = self.dropout(self.encoder(z))
        return self.head(z.mean(dim=1)).squeeze(-1).reshape(B, C)


def load_model(L, h):
    ckpt = torch.load(MODEL_DIR / f'L{L}_h{h}_best.pt',
                      map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    m = PatchTSTv2(
        L=cfg['L'], patch_len=cfg['patch_len'], stride=cfg['stride'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'],
        dropout=cfg['dropout'], n_assets=cfg['n_assets']).to(DEVICE)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


# ── Generate Predictions (2019–2024) ──────────────────────────────────

L, h = 252, 21
model = load_model(L, h)
eval_start = date_to_idx('2019-01-01')
eval_end = len(dates) - 1

print(f"Generating predictions: {dates[eval_start].date()} to "
      f"{dates[eval_end].date()}")

pred_all, actual_all, dates_all, mkt_ret_all = [], [], [], []
with torch.no_grad():
    for t in range(max(eval_start, L), eval_end - h + 2):
        X = scaled_filled[t - L:t, :].T[np.newaxis, :, :]
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        pred_s = model(X_t).cpu().numpy()[0]
        v = vol_vals[t, :]
        pred_raw = pred_s * v
        actual_raw = np.nansum(log_ret_vals[t:t + h, :], axis=0)
        mkt = np.nanmean(actual_raw)
        pred_all.append(pred_raw)
        actual_all.append(actual_raw)
        dates_all.append(dates[t])
        mkt_ret_all.append(mkt)

pred_all = np.array(pred_all)
actual_all = np.array(actual_all)
dates_arr = pd.DatetimeIndex(dates_all)
mkt_ret = np.array(mkt_ret_all)

del model
torch.cuda.empty_cache()
print(f"Total prediction windows: {len(dates_arr)}")


# ── Cross-Sectional IC & Hit Rate ─────────────────────────────────────

cs_ic, cs_hit = [], []
for w in range(len(dates_arr)):
    p, a = pred_all[w], actual_all[w]
    v = ~(np.isnan(p) | np.isnan(a))
    if v.sum() < 20:
        cs_ic.append(np.nan)
        cs_hit.append(np.nan)
        continue
    c, _ = stats.spearmanr(p[v], a[v])
    cs_ic.append(c)
    cs_hit.append((np.sign(p[v]) == np.sign(a[v])).mean())

cs_ic = np.array(cs_ic)
cs_hit = np.array(cs_hit)


# ── Regime Figure ─────────────────────────────────────────────────────

roll_ic = pd.Series(cs_ic, index=dates_arr).rolling(63, min_periods=21).mean()
mkt_roll = pd.Series(mkt_ret, index=dates_arr).rolling(63, min_periods=21).mean()
roll_hit = pd.Series(cs_hit, index=dates_arr).rolling(63, min_periods=21).mean()

regimes = [
    ('2019-01-01', '2020-02-19', 'Pre-COVID\nBull',    'green'),
    ('2020-02-20', '2020-03-23', 'COVID\nCrash',       'red'),
    ('2020-03-24', '2020-12-31', 'Recovery\nRally',     'green'),
    ('2021-01-01', '2021-12-31', '2021\nBull',          'blue'),
    ('2022-01-01', '2022-10-12', '2022 Rate\nHiking',   'red'),
    ('2022-10-13', '2024-12-31', '2023-24\nRecovery',   'green'),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel (a): Rolling IC
ax = axes[0]
ax.plot(roll_ic.index, roll_ic.values, lw=1.2, color='steelblue')
ax.axhline(0, ls='-', c='black', lw=0.5)
ax.fill_between(roll_ic.index, roll_ic.values, 0,
                where=roll_ic.values > 0, alpha=0.3, color='green')
ax.fill_between(roll_ic.index, roll_ic.values, 0,
                where=roll_ic.values < 0, alpha=0.3, color='red')
ax.set_ylabel('Rolling 63-day IC\n(Spearman)')
ax.set_title('(a) Cross-Sectional Predictive Signal Over Time (L252, h21)')

for s, e, lbl, clr in regimes:
    s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
    ax.axvspan(s_ts, e_ts, alpha=0.08, color=clr)
    mid = s_ts + (e_ts - s_ts) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.85, lbl, ha='center',
            fontsize=7, alpha=0.7)

# Panel (b): Market return
ax = axes[1]
ax.plot(mkt_roll.index, mkt_roll.values * 100, lw=1.2, color='#d7191c')
ax.axhline(0, ls='-', c='black', lw=0.5)
ax.fill_between(mkt_roll.index, mkt_roll.values * 100, 0,
                where=mkt_roll.values > 0, alpha=0.2, color='green')
ax.fill_between(mkt_roll.index, mkt_roll.values * 100, 0,
                where=mkt_roll.values < 0, alpha=0.2, color='red')
ax.set_ylabel('Rolling 63-day\nAvg Return (%)')
ax.set_title('(b) Market Regime (Cross-Sectional Average '
             'Cumulative 21-day Return)')
for s, e, lbl, clr in regimes:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color=clr)

# Panel (c): Hit rate
ax = axes[2]
ax.plot(roll_hit.index, roll_hit.values, lw=1.2, color='#fdae61')
ax.axhline(0.5, ls='--', c='black', lw=0.8, label='50% (coin flip)')
ax.set_ylabel('Rolling 63-day\nHit Rate')
ax.set_title('(c) Directional Accuracy Over Time')
ax.legend(fontsize=8)
for s, e, lbl, clr in regimes:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color=clr)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()
fig.savefig(OUT / 'fig_regime_analysis.png', dpi=FIG_DPI)
plt.close()
print("-> fig_regime_analysis.png")


# ── Regime Table ──────────────────────────────────────────────────────

periods = [
    ('Pre-COVID Bull',   '2019-01-01', '2020-02-19'),
    ('COVID Crash',      '2020-02-20', '2020-03-23'),
    ('Recovery Rally',   '2020-03-24', '2020-12-31'),
    ('2021 Bull',        '2021-01-01', '2021-12-31'),
    ('2022 Rate Hiking', '2022-01-01', '2022-10-12'),
    ('2023-24 Recovery', '2022-10-13', '2024-12-31'),
    ('Full Val (2019-20)',  '2019-01-01', '2020-12-31'),
    ('Full Test (2021-24)', '2021-01-01', '2024-12-31'),
]

period_rows = []
for name, s, e in periods:
    mask = (dates_arr >= pd.Timestamp(s)) & (dates_arr <= pd.Timestamp(e))
    if mask.sum() < 10:
        continue
    p_sub, a_sub = pred_all[mask], actual_all[mask]
    ic_sub, hit_sub = cs_ic[mask], cs_hit[mask]
    valid = ~(np.isnan(p_sub) | np.isnan(a_sub))
    se = (p_sub - a_sub) ** 2
    se[~valid] = np.nan
    mse = np.nanmean(se)
    period_rows.append({
        'Period': name, 'Start': s, 'End': e,
        'N Windows': mask.sum(),
        'Mean CS IC': np.nanmean(ic_sub),
        'Median CS IC': np.nanmedian(ic_sub),
        'Pct IC > 0': (ic_sub[~np.isnan(ic_sub)] > 0).mean() * 100,
        'Mean Hit Rate': np.nanmean(hit_sub),
        'MSE': mse,
        'Avg Mkt Ret': np.nanmean(mkt_ret[mask]) * 100,
    })

period_table = pd.DataFrame(period_rows)
period_table.to_csv(OUT / 'table_regime_analysis.csv', index=False)
print("\nRegime Analysis:")
print(period_table[['Period', 'N Windows', 'Mean CS IC',
                    'Mean Hit Rate', 'MSE']].to_string(index=False))
