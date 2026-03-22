"""
Model Evaluation
Evaluates all 9 PatchTST models against benchmarks (Random Walk, AR(1),
Historical Mean) on MSE, directional accuracy, and Information Coefficient.
Also runs Carhart four-factor spanning regressions.
Requires: preprocessed_data.pkl, trained models in models_v2/
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.multitest import multipletests
import pandas_datareader.data as web
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

RESULTS_DIR = Path('results_v2')
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path('models_v2')

LOOKBACKS = [63, 126, 252]
HORIZONS = [5, 21, 63]
SEED = 42
np.random.seed(SEED)

# ── Data Loading ──────────────────────────────────────────────────────

print("Loading data...")
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
mask_valid = ~np.isnan(scaled_vals)
scaled_filled = np.nan_to_num(scaled_vals, nan=0.0)


def date_to_idx(d):
    return min(dates.searchsorted(pd.Timestamp(d)), len(dates) - 1)


test_start = date_to_idx(splits['test'][0])
test_end = len(dates) - 1
print(f"Test: {dates[test_start].date()} to {dates[test_end].date()}")


# ── Model Architecture (must match training) ──────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, L, patch_len, stride, d_model):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = int((L - patch_len) / stride) + 1
        self.pad_len = max(0, (self.n_patches - 1) * stride + patch_len - L)
        self.padding = nn.ReplicationPad1d((0, self.pad_len))
        L_padded = L + self.pad_len
        self.n_patches = int((L_padded - patch_len) / stride) + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B * C, 1, L)
        x = self.padding(x).squeeze(1)
        x = x.unfold(1, self.patch_len, self.stride)
        x = self.proj(x)
        x = x + self.pos_enc
        return x, B, C


class PatchTSTv2(nn.Module):
    def __init__(self, L, patch_len, stride, d_model, n_heads,
                 n_layers, d_ff, dropout, n_assets):
        super().__init__()
        self.patch_embed = PatchEmbedding(L, patch_len, stride, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L = x.shape
        z, _, _ = self.patch_embed(x)
        z = self.encoder(z)
        z = self.dropout(z)
        z = z.mean(dim=1)
        out = self.head(z).squeeze(-1)
        return out.reshape(B, C)


def load_model(L, h):
    ckpt = torch.load(MODEL_DIR / f'L{L}_h{h}_best.pt',
                      map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    model = PatchTSTv2(
        L=cfg['L'], patch_len=cfg['patch_len'], stride=cfg['stride'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'],
        dropout=cfg['dropout'], n_assets=cfg['n_assets']
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# ── Prediction Generation ─────────────────────────────────────────────

def generate_predictions(model, L, h):
    model.eval()
    preds, actuals, wdates = [], [], []
    with torch.no_grad():
        for t in range(max(test_start, L), test_end - h + 2):
            X = scaled_filled[t - L:t, :].T[np.newaxis, :, :]
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            pred_scaled = model(X_t).cpu().numpy()[0]
            v = vol_vals[t, :]
            pred_raw = pred_scaled * v
            actual_raw = np.nansum(log_ret_vals[t:t + h, :], axis=0)
            preds.append(pred_raw)
            actuals.append(actual_raw)
            wdates.append(t)
    return np.array(preds), np.array(actuals), wdates


# ── Benchmark Predictions ─────────────────────────────────────────────

def random_walk_preds(h):
    preds, actuals, wd = [], [], []
    for t in range(test_start, test_end - h + 2):
        preds.append(np.zeros(n_assets))
        actuals.append(np.nansum(log_ret_vals[t:t + h, :], axis=0))
        wd.append(t)
    return np.array(preds), np.array(actuals), wd


def ar1_preds(L, h):
    preds, actuals, wd = [], [], []
    for t in range(max(test_start, L), test_end - h + 2):
        pred = np.zeros(n_assets)
        for i in range(n_assets):
            w = log_ret_vals[t - L:t, i]
            v = w[~np.isnan(w)]
            if len(v) > 10 and np.std(v[:-1]) > 1e-10:
                phi = (np.corrcoef(v[:-1], v[1:])[0, 1] *
                       np.std(v[1:]) / np.std(v[:-1]))
                last = v[-1]
                cum = 0
                for s in range(h):
                    last = phi * last
                    cum += last
                pred[i] = cum
        preds.append(pred)
        actuals.append(np.nansum(log_ret_vals[t:t + h, :], axis=0))
        wd.append(t)
    return np.array(preds), np.array(actuals), wd


def hist_mean_preds(h):
    preds, actuals, wd = [], [], []
    for t in range(test_start, test_end - h + 2):
        pred = np.zeros(n_assets)
        for i in range(n_assets):
            hist = log_ret_vals[:t, i]
            v = hist[~np.isnan(hist)]
            if len(v) > 0:
                pred[i] = np.mean(v) * h
        preds.append(pred)
        actuals.append(np.nansum(log_ret_vals[t:t + h, :], axis=0))
        wd.append(t)
    return np.array(preds), np.array(actuals), wd


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(pred_cum, actual_cum):
    valid = ~(np.isnan(pred_cum) | np.isnan(actual_cum))

    se = (pred_cum - actual_cum) ** 2
    se[~valid] = np.nan
    mse_overall = np.nanmean(np.nanmean(se, axis=0))

    correct = (np.sign(pred_cum) == np.sign(actual_cum)).astype(float)
    correct[~valid] = np.nan
    hit_overall = np.nanmean(np.nanmean(correct, axis=0))

    n_ast = pred_cum.shape[1]
    ic_arr = np.full(n_ast, np.nan)
    ic_pvals = np.full(n_ast, np.nan)
    for i in range(n_ast):
        v = valid[:, i]
        if v.sum() < 30:
            continue
        p, a = pred_cum[v, i], actual_cum[v, i]
        if np.std(p) < 1e-10 or np.std(a) < 1e-10:
            continue
        c, pv = stats.pearsonr(p, a)
        ic_arr[i] = c
        ic_pvals[i] = pv

    ic_mean = np.nanmean(ic_arr)
    ic_median = np.nanmedian(ic_arr)

    vp = ic_pvals[~np.isnan(ic_pvals)]
    if len(vp) > 0:
        rej, _, _, _ = multipletests(vp, alpha=0.05, method='fdr_bh')
        n_sig, pct_sig = rej.sum(), rej.mean() * 100
    else:
        n_sig, pct_sig = 0, 0.0

    return {
        'MSE': mse_overall, 'Hit_Rate': hit_overall,
        'IC_mean': ic_mean, 'IC_median': ic_median,
        'IC_per_asset': ic_arr, 'IC_pvals': ic_pvals,
        'IC_n_sig_BH': n_sig, 'IC_pct_sig_BH': pct_sig,
        'N_eval': np.sum(~np.isnan(ic_arr)),
    }


# ── Main Evaluation Loop ─────────────────────────────────────────────

if __name__ == '__main__':
    all_results = {}
    for L in LOOKBACKS:
        for h in HORIZONS:
            c = f'L{L}_h{h}'
            print(f"\n--- {c} ---")

            model = load_model(L, h)
            pr, ar, wd = generate_predictions(model, L, h)
            ptst = compute_metrics(pr, ar)
            print(f"  PatchTST: MSE={ptst['MSE']:.6f}, "
                  f"Hit={ptst['Hit_Rate']:.4f}, IC={ptst['IC_mean']:.4f}")

            rw_p, rw_a, rw_w = random_walk_preds(h)
            rw_start = len(rw_w) - len(wd)
            rw = compute_metrics(rw_p[rw_start:], rw_a[rw_start:])

            ar_p, ar_a, ar_w = ar1_preds(L, h)
            ar1 = compute_metrics(ar_p, ar_a)

            hm_p, hm_a, hm_w = hist_mean_preds(h)
            hm_start = len(hm_w) - len(wd)
            hm = compute_metrics(hm_p[hm_start:], hm_a[hm_start:])

            all_results[c] = {
                'PatchTST': ptst, 'RW': rw, 'AR1': ar1, 'HistMean': hm,
                'pred_raw': pr, 'actual_raw': ar, 'window_dates': wd,
            }
            del model
            torch.cuda.empty_cache()

    # Save summary tables
    rows = []
    for L in LOOKBACKS:
        for h in HORIZONS:
            c = f'L{L}_h{h}'
            r = all_results[c]
            rows.append({
                'Config': c, 'L': L, 'h': h,
                'PatchTST': r['PatchTST']['MSE'],
                'Random Walk': r['RW']['MSE'],
                'AR(1)': r['AR1']['MSE'],
                'Hist Mean': r['HistMean']['MSE'],
            })
    mse_tbl = pd.DataFrame(rows)
    mse_tbl.to_csv(RESULTS_DIR / 'table_mse.csv', index=False)
    print("\nMSE Table:")
    print(mse_tbl.to_string(index=False))

    # Factor analysis
    try:
        ff = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench',
                            start='2021-01-01', end='2024-12-31')[0] / 100
        mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench',
                             start='2021-01-01', end='2024-12-31')[0] / 100
        ff.index = pd.to_datetime(ff.index)
        mom.index = pd.to_datetime(mom.index)
        factors = ff.join(mom, how='inner')
        factors.columns = ['MKT', 'SMB', 'HML', 'RF', 'UMD']
        print(f"\nFF factors loaded: {len(factors)} days")

        fa_rows = []
        for L in LOOKBACKS:
            for h in HORIZONS:
                c = f'L{L}_h{h}'
                r = all_results[c]
                pr, ar, wd = r['pred_raw'], r['actual_raw'], r['window_dates']

                ls_rets, ls_dates = [], []
                for w in range(len(wd)):
                    p, a = pr[w], ar[w]
                    v = ~(np.isnan(p) | np.isnan(a))
                    if v.sum() < 40:
                        continue
                    vi = np.where(v)[0]
                    ranks = np.argsort(p[vi])
                    nq = len(ranks) // 5
                    ls = (np.mean(a[vi[ranks[-nq:]]]) -
                          np.mean(a[vi[ranks[:nq]]]))
                    ls_rets.append(ls)
                    ls_dates.append(dates[wd[w]])

                if len(ls_rets) < 30:
                    continue

                ls_s = pd.Series(ls_rets, index=pd.DatetimeIndex(ls_dates))
                ls_s = ls_s.iloc[::h]

                reg = []
                for dt in ls_s.index:
                    loc = factors.index.searchsorted(dt)
                    if loc + h > len(factors):
                        continue
                    fw = factors.iloc[loc:loc + h]
                    reg.append({
                        'LS': ls_s[dt], 'MKT': fw['MKT'].sum(),
                        'SMB': fw['SMB'].sum(), 'HML': fw['HML'].sum(),
                        'UMD': fw['UMD'].sum(), 'RF': fw['RF'].sum(),
                    })

                rd = pd.DataFrame(reg)
                if len(rd) < 20:
                    continue

                rd['LS_ex'] = rd['LS'] - rd['RF']
                Y = rd['LS_ex']
                X = add_constant(rd[['MKT', 'SMB', 'HML', 'UMD']])
                ols = OLS(Y, X).fit(cov_type='HC1')

                fa_rows.append({
                    'Config': c, 'Alpha': ols.params['const'],
                    'Alpha_t': ols.tvalues['const'],
                    'Alpha_p': ols.pvalues['const'],
                    'Beta_MKT': ols.params['MKT'],
                    'Beta_UMD': ols.params['UMD'],
                    'R2': ols.rsquared,
                })

        if fa_rows:
            fa_tbl = pd.DataFrame(fa_rows)
            fa_tbl.to_csv(RESULTS_DIR / 'table_factors.csv', index=False)
            print("\nFactor Analysis:")
            print(fa_tbl.to_string(index=False))

    except Exception as e:
        print(f"FF factor download failed: {e}")

    with open(RESULTS_DIR / 'all_predictions.pkl', 'wb') as f:
        pickle.dump({c: {'pred': r['pred_raw'], 'actual': r['actual_raw'],
                         'dates': r['window_dates']}
                     for c, r in all_results.items()}, f)

    print(f"\nAll results saved to ./{RESULTS_DIR}/")
