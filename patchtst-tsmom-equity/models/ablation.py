"""
Ablation Study and Multi-Horizon Ensembling
Part A: Ablation over depth, patch length, dropout, and positional encoding
Part B: Multi-horizon ensemble (equal-weight and validation-optimised)
All evaluated on L=252, h=21 configuration.
Requires: preprocessed_data.pkl, trained base models in models_v2/
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

ABL_DIR = Path('models_ablation')
ABL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path('results_ablation')
RESULTS_DIR.mkdir(exist_ok=True)
V2_DIR = Path('models_v2')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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


train_end = date_to_idx(splits['train'][1])
val_start = date_to_idx(splits['val'][0])
val_end = date_to_idx(splits['val'][1])
test_start = date_to_idx(splits['test'][0])
test_end = len(dates) - 1


# ── Dataset ───────────────────────────────────────────────────────────

class TSMOMDatasetV2(Dataset):
    def __init__(self, data, mask, start_idx, end_idx, L, h):
        first = max(start_idx, L)
        last = end_idx - h + 1
        idx = np.arange(first, last)
        x_idx = idx[:, None] + np.arange(-L, 0)[None, :]
        y_idx = idx[:, None] + np.arange(0, h)[None, :]
        X = data[x_idx].transpose(0, 2, 1)
        M = mask[x_idx].transpose(0, 2, 1).astype(np.float32)
        Y_seq = data[y_idx].transpose(0, 2, 1)
        Y_cum = np.nansum(Y_seq, axis=2)
        tv = (~np.isnan(data[y_idx].transpose(0, 2, 1))).mean(axis=2)
        Y_cum[tv < 0.5] = 0.0
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y_cum, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.tv = torch.tensor((tv >= 0.5).astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.M[i], self.tv[i]


# ── Model (with optional positional encoding) ────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, L, patch_len, stride, d_model, use_pos_enc=True):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.use_pos_enc = use_pos_enc
        self.n_patches = int((L - patch_len) / stride) + 1
        self.pad_len = max(0, (self.n_patches - 1) * stride + patch_len - L)
        self.padding = nn.ReplicationPad1d((0, self.pad_len))
        self.n_patches = int((L + self.pad_len - patch_len) / stride) + 1
        self.proj = nn.Linear(patch_len, d_model)
        if use_pos_enc:
            self.pos_enc = nn.Parameter(
                torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B * C, 1, L)
        x = self.padding(x).squeeze(1)
        x = x.unfold(1, self.patch_len, self.stride)
        x = self.proj(x)
        if self.use_pos_enc:
            x = x + self.pos_enc
        return x, B, C


class PatchTSTAblation(nn.Module):
    def __init__(self, L, patch_len, stride, d_model, n_heads,
                 n_layers, d_ff, dropout, n_assets, use_pos_enc=True):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            L, patch_len, stride, d_model, use_pos_enc)
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


# ── Training & Evaluation Utilities ───────────────────────────────────

def masked_mse(pred, target, mask_lb, mask_tgt):
    active = (mask_lb.mean(dim=2) > 0.5).float() * mask_tgt
    se = (pred - target) ** 2
    return (se * active).sum() / (active.sum() + 1e-8)


def get_lr(epoch, warmup, max_ep, base_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    return base_lr * 0.5 * (1 + np.cos(
        np.pi * (epoch - warmup) / (max_ep - warmup)))


def train_ablation(name, L, h, patch_len, stride, d_model, n_heads,
                   n_layers, d_ff, dropout, use_pos_enc=True,
                   lr=1e-4, epochs=200, patience=20, warmup=10):
    print(f"\n  Training: {name}")
    train_ds = TSMOMDatasetV2(scaled_filled, mask_valid,
                              0, train_end + 1, L, h)
    val_ds = TSMOMDatasetV2(scaled_filled, mask_valid,
                            val_start, val_end + 1, L, h)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False,
                        num_workers=0, pin_memory=True)

    model = PatchTSTAblation(
        L=L, patch_len=patch_len, stride=stride, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        n_assets=n_assets, use_pos_enc=use_pos_enc).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-5)
    best_val, pat_ctr = float('inf'), 0

    for epoch in range(epochs):
        cur_lr = get_lr(epoch, warmup, epochs, lr)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        model.train()
        for X, Y, M, Mv in train_dl:
            X, Y, M, Mv = (X.to(DEVICE), Y.to(DEVICE),
                            M.to(DEVICE), Mv.to(DEVICE))
            optimizer.zero_grad()
            loss = masked_mse(model(X), Y, M, Mv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for X, Y, M, Mv in val_dl:
                X, Y, M, Mv = (X.to(DEVICE), Y.to(DEVICE),
                                M.to(DEVICE), Mv.to(DEVICE))
                vl.append(masked_mse(model(X), Y, M, Mv).item())

        avg_v = np.mean(vl)
        if avg_v < best_val:
            best_val = avg_v
            pat_ctr = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'L': L, 'h': h, 'patch_len': patch_len,
                    'stride': stride, 'd_model': d_model,
                    'n_heads': n_heads, 'n_layers': n_layers,
                    'd_ff': d_ff, 'dropout': dropout,
                    'n_assets': n_assets, 'use_pos_enc': use_pos_enc}
            }, ABL_DIR / f'{name}_best.pt')
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                break

    print(f"    Best val: {best_val:.6f}")
    del model
    torch.cuda.empty_cache()
    return best_val


def evaluate_model(name, L, h):
    ckpt = torch.load(ABL_DIR / f'{name}_best.pt',
                      map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    model = PatchTSTAblation(
        L=cfg['L'], patch_len=cfg['patch_len'], stride=cfg['stride'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'],
        dropout=cfg['dropout'], n_assets=cfg['n_assets'],
        use_pos_enc=cfg.get('use_pos_enc', True)).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds, actuals = [], []
    with torch.no_grad():
        for t in range(max(test_start, L), test_end - h + 2):
            X = scaled_filled[t - L:t, :].T[np.newaxis, :, :]
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            p_s = model(X_t).cpu().numpy()[0]
            preds.append(p_s * vol_vals[t, :])
            actuals.append(np.nansum(log_ret_vals[t:t + h, :], axis=0))

    preds, actuals = np.array(preds), np.array(actuals)
    valid = ~(np.isnan(preds) | np.isnan(actuals))

    se = (preds - actuals) ** 2
    se[~valid] = np.nan
    mse = np.nanmean(se)

    hits = (np.sign(preds) == np.sign(actuals)).astype(float)
    hits[~valid] = np.nan
    hit = np.nanmean(hits)

    ic_arr = []
    for i in range(n_assets):
        v = valid[:, i]
        if v.sum() < 30:
            continue
        c, _ = stats.pearsonr(preds[v, i], actuals[v, i])
        ic_arr.append(c)

    del model
    torch.cuda.empty_cache()
    return {'MSE': mse, 'Hit': hit, 'IC': np.mean(ic_arr) if ic_arr else np.nan}


# ══════════════════════════════════════════════════════════════════════
# PART A: ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("PART A: ABLATION STUDY (L=252, h=21)")
    print("=" * 60)

    L, h = 252, 21
    BASE = {
        'L': L, 'h': h, 'd_model': 64, 'n_heads': 4, 'd_ff': 128,
        'patch_len': 16, 'stride': 8, 'n_layers': 3, 'dropout': 0.3,
        'use_pos_enc': True
    }

    ablations = {'baseline': {**BASE}}

    for nl in [1, 2]:
        ablations[f'layers_{nl}'] = {**BASE, 'n_layers': nl}
    for pl in [8, 32]:
        ablations[f'patch_{pl}'] = {**BASE, 'patch_len': pl,
                                    'stride': max(pl // 2, 1)}
    ablations['no_pos_enc'] = {**BASE, 'use_pos_enc': False}
    for dr in [0.1, 0.2, 0.5]:
        ablations[f'dropout_{dr}'] = {**BASE, 'dropout': dr}

    abl_results = {}
    for name, cfg in ablations.items():
        val_loss = train_ablation(
            name=name, L=cfg['L'], h=cfg['h'],
            patch_len=cfg['patch_len'], stride=cfg['stride'],
            d_model=cfg['d_model'], n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'], d_ff=cfg['d_ff'],
            dropout=cfg['dropout'],
            use_pos_enc=cfg.get('use_pos_enc', True))
        test_metrics = evaluate_model(name, L, h)
        abl_results[name] = {'val_loss': val_loss, **test_metrics,
                             'config': cfg}
        print(f"    Test: MSE={test_metrics['MSE']:.6f}, "
              f"Hit={test_metrics['Hit']:.4f}, IC={test_metrics['IC']:.4f}")

    abl_rows = []
    for name, r in abl_results.items():
        cfg = r['config']
        abl_rows.append({
            'Variant': name, 'Layers': cfg['n_layers'],
            'Patch': cfg['patch_len'], 'Dropout': cfg['dropout'],
            'PosEnc': cfg.get('use_pos_enc', True),
            'Val Loss': r['val_loss'], 'Test MSE': r['MSE'],
            'Hit Rate': r['Hit'], 'IC Mean': r['IC'],
        })
    abl_table = pd.DataFrame(abl_rows)
    abl_table.to_csv(RESULTS_DIR / 'table_ablation.csv', index=False)
    print("\nAblation Results:")
    print(abl_table.to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # PART B: MULTI-HORIZON ENSEMBLE
    # ══════════════════════════════════════════════════════════════════

    print("\n\n" + "=" * 60)
    print("PART B: MULTI-HORIZON ENSEMBLE")
    print("=" * 60)

    # Re-use PatchTSTv2 from models/train.py for loading base models
    class PatchTSTv2(nn.Module):
        def __init__(self, L, patch_len, stride, d_model, n_heads,
                     n_layers, d_ff, dropout, n_assets):
            super().__init__()
            self.patch_embed = PatchEmbedding(
                L, patch_len, stride, d_model, True)
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

    def load_v2(L, h):
        ckpt = torch.load(V2_DIR / f'L{L}_h{h}_best.pt',
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

    def get_predictions(model, L):
        model.eval()
        preds, actuals, wdates = [], [], []
        with torch.no_grad():
            for t in range(max(test_start, L), test_end - 21 + 2):
                X = scaled_filled[t - L:t, :].T[np.newaxis, :, :]
                X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
                p_s = model(X_t).cpu().numpy()[0]
                preds.append(p_s * vol_vals[t, :])
                actuals.append(
                    np.nansum(log_ret_vals[t:t + 21, :], axis=0))
                wdates.append(t)
        return np.array(preds), np.array(actuals), wdates

    horizons_to_ensemble = [5, 21, 63]
    horizon_preds = {}
    for h_model in horizons_to_ensemble:
        m = load_v2(252, h_model)
        p, a, w = get_predictions(m, 252)
        horizon_preds[h_model] = p
        if h_model == 21:
            ensemble_actuals = a
        del m
        torch.cuda.empty_cache()

    min_len = min(len(horizon_preds[h]) for h in horizons_to_ensemble)
    for h in horizons_to_ensemble:
        horizon_preds[h] = horizon_preds[h][:min_len]
    ensemble_actuals = ensemble_actuals[:min_len]

    # Equal-weight ensemble
    ew_pred = np.mean([horizon_preds[h] for h in horizons_to_ensemble],
                      axis=0)
    valid = ~(np.isnan(ew_pred) | np.isnan(ensemble_actuals))
    se = (ew_pred - ensemble_actuals) ** 2
    se[~valid] = np.nan
    ew_mse = np.nanmean(se)
    hits = (np.sign(ew_pred) == np.sign(ensemble_actuals)).astype(float)
    hits[~valid] = np.nan
    ew_hit = np.nanmean(hits)
    print(f"\nEqual-Weight: MSE={ew_mse:.6f}, Hit={ew_hit:.4f}")

    # Validation-optimised weights via grid search
    val_preds = {}
    for h_model in horizons_to_ensemble:
        m = load_v2(252, h_model)
        vp = []
        with torch.no_grad():
            for t in range(max(val_start, 252), val_end - 21 + 2):
                X = scaled_filled[t - 252:t, :].T[np.newaxis, :, :]
                X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
                p_s = m(X_t).cpu().numpy()[0]
                vp.append(p_s * vol_vals[t, :])
        val_preds[h_model] = np.array(vp)
        del m
        torch.cuda.empty_cache()

    val_min = min(len(val_preds[h]) for h in horizons_to_ensemble)
    for h in horizons_to_ensemble:
        val_preds[h] = val_preds[h][:val_min]

    # Generate val actuals
    val_actuals = []
    for t in range(max(val_start, 252), val_end - 21 + 2):
        val_actuals.append(np.nansum(log_ret_vals[t:t + 21, :], axis=0))
    val_actuals = np.array(val_actuals)[:val_min]

    best_ic, best_w = -np.inf, (0.0, 1.0, 0.0)
    for w5 in np.arange(0, 1.05, 0.1):
        for w21 in np.arange(0, 1.05 - w5, 0.1):
            w63 = round(1.0 - w5 - w21, 1)
            if w63 < -0.01:
                continue
            combo = (w5 * val_preds[5] + w21 * val_preds[21] +
                     w63 * val_preds[63])
            valid_v = ~(np.isnan(combo) | np.isnan(val_actuals))
            ics = []
            for i in range(n_assets):
                v = valid_v[:, i]
                if v.sum() < 20:
                    continue
                c, _ = stats.pearsonr(combo[v, i], val_actuals[v, i])
                ics.append(c)
            avg_ic = np.mean(ics) if ics else -np.inf
            if avg_ic > best_ic:
                best_ic = avg_ic
                best_w = (w5, w21, w63)

    print(f"Optimised weights: h5={best_w[0]:.1f}, h21={best_w[1]:.1f}, "
          f"h63={best_w[2]:.1f}")

    opt_pred = (best_w[0] * horizon_preds[5] +
                best_w[1] * horizon_preds[21] +
                best_w[2] * horizon_preds[63])
    valid = ~(np.isnan(opt_pred) | np.isnan(ensemble_actuals))
    se = (opt_pred - ensemble_actuals) ** 2
    se[~valid] = np.nan
    opt_mse = np.nanmean(se)
    hits = (np.sign(opt_pred) == np.sign(ensemble_actuals)).astype(float)
    hits[~valid] = np.nan
    opt_hit = np.nanmean(hits)
    print(f"Optimised: MSE={opt_mse:.6f}, Hit={opt_hit:.4f}")

    # Summary table
    ens_rows = []
    for h_model in horizons_to_ensemble:
        hp = horizon_preds[h_model]
        valid = ~(np.isnan(hp) | np.isnan(ensemble_actuals))
        se = (hp - ensemble_actuals) ** 2
        se[~valid] = np.nan
        hits = (np.sign(hp) == np.sign(ensemble_actuals)).astype(float)
        hits[~valid] = np.nan
        ens_rows.append({
            'Method': f'L252_h{h_model} (single)',
            'MSE': np.nanmean(se), 'Hit': np.nanmean(hits)})
    ens_rows.append({'Method': 'Equal-Weight Ensemble',
                     'MSE': ew_mse, 'Hit': ew_hit})
    ens_rows.append({'Method': f'Optimised (w={best_w})',
                     'MSE': opt_mse, 'Hit': opt_hit})

    ens_table = pd.DataFrame(ens_rows)
    ens_table.to_csv(RESULTS_DIR / 'table_ensemble.csv', index=False)
    print("\nEnsemble Comparison:")
    print(ens_table.to_string(index=False))
    print(f"\nAll results saved to ./{RESULTS_DIR}/")
