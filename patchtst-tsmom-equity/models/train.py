"""
PatchTST Model Training
Trains 9 PatchTST models across 3 lookback windows x 3 forecast horizons
for time-series momentum forecasting.
Requires: preprocessed_data.pkl, NVIDIA GPU recommended
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import json
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

MODEL_DIR = Path('models_v2')
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Configuration ─────────────────────────────────────────────────────

LOOKBACKS = [63, 126, 252]
HORIZONS = [5, 21, 63]

BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 10

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128
DROPOUT = 0.3
PATCH_LEN = 16
STRIDE = 8

# ── Data Loading ──────────────────────────────────────────────────────

print("Loading preprocessed data...")
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

scaled_wide = data['scaled_returns']
splits = data['splits']
dates = scaled_wide.index

scaled_vals = scaled_wide.to_numpy(dtype=np.float64, na_value=np.nan)
mask_valid = ~np.isnan(scaled_vals)
scaled_vals_filled = np.nan_to_num(scaled_vals, nan=0.0)
n_assets = scaled_vals.shape[1]


def date_to_idx(d):
    return min(dates.searchsorted(pd.Timestamp(d)), len(dates) - 1)


train_end = date_to_idx(splits['train'][1])
val_start = date_to_idx(splits['val'][0])
val_end = date_to_idx(splits['val'][1])

print(f"Data shape: {scaled_vals.shape}")
print(f"Train: 0–{train_end}, Val: {val_start}–{val_end}")


# ── Dataset ───────────────────────────────────────────────────────────

class TSMOMDatasetV2(Dataset):
    """
    X: (C, L)  — lookback window of scaled returns
    Y: (C,)    — scalar cumulative h-step scaled return
    M: (C, L)  — validity mask
    """
    def __init__(self, data, mask, start_idx, end_idx, L, h):
        first = max(start_idx, L)
        last = end_idx - h + 1
        n = last - first
        print(f"    Building {n} windows (L={L}, h={h})...")

        idx = np.arange(first, last)
        x_idx = idx[:, None] + np.arange(-L, 0)[None, :]
        y_idx = idx[:, None] + np.arange(0, h)[None, :]

        X = data[x_idx].transpose(0, 2, 1)
        M = mask[x_idx].transpose(0, 2, 1).astype(np.float32)
        Y_seq = data[y_idx].transpose(0, 2, 1)
        Y_cum = np.nansum(Y_seq, axis=2)

        target_valid = (~np.isnan(data[y_idx].transpose(0, 2, 1))).mean(axis=2)
        Y_cum[target_valid < 0.5] = 0.0

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y_cum, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)
        self.target_valid = torch.tensor((target_valid >= 0.5).astype(np.float32))
        print(f"    Done. X: {self.X.shape}, Y: {self.Y.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.M[idx], self.target_valid[idx]


# ── Model Architecture ────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, L, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
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
        x = self.padding(x)
        x = x.squeeze(1)
        x = x.unfold(1, self.patch_len, self.stride)
        x = self.proj(x)
        x = x + self.pos_enc
        return x, B, C


class PatchTSTv2(nn.Module):
    """
    PatchTST with mean-pooling head -> scalar output per channel.
    Input:  (B, C, L)
    Output: (B, C) — predicted cumulative h-step return per asset
    """
    def __init__(self, L, patch_len, stride, d_model, n_heads,
                 n_layers, d_ff, dropout, n_assets):
        super().__init__()
        self.patch_embed = PatchEmbedding(L, patch_len, stride, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L = x.shape
        z, _, _ = self.patch_embed(x)
        z = self.encoder(z)
        z = self.dropout(z)
        z = z.mean(dim=1)
        out = self.head(z).squeeze(-1)
        return out.reshape(B, C)


# ── Training Utilities ────────────────────────────────────────────────

def masked_mse_loss(pred, target, mask_lb, mask_tgt):
    lb_valid = (mask_lb.mean(dim=2) > 0.5).float()
    active = lb_valid * mask_tgt
    se = (pred - target) ** 2
    masked = se * active
    return masked.sum() / (active.sum() + 1e-8)


def get_lr(epoch, warmup, max_epochs, base_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / (max_epochs - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train_model(L, h):
    config = f"L{L}_h{h}"
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {config} (lookback={L}d, horizon={h}d)")
    print(f"{'=' * 60}")

    train_ds = TSMOMDatasetV2(scaled_vals_filled, mask_valid,
                              0, train_end + 1, L, h)
    val_ds = TSMOMDatasetV2(scaled_vals_filled, mask_valid,
                            val_start, val_end + 1, L, h)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")

    p_len = min(PATCH_LEN, L // 4)
    s = max(p_len // 2, 1)

    model = PatchTSTv2(
        L=L, patch_len=p_len, stride=s,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, n_assets=n_assets
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Patch: len={p_len}, stride={s}, "
          f"n_patches={model.patch_embed.n_patches}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)

    best_val = float('inf')
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(EPOCHS):
        t0 = time.time()
        lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Train
        model.train()
        tl = []
        for X, Y, M, Mv in train_dl:
            X, Y, M, Mv = (X.to(DEVICE), Y.to(DEVICE),
                            M.to(DEVICE), Mv.to(DEVICE))
            optimizer.zero_grad()
            pred = model(X)
            loss = masked_mse_loss(pred, Y, M, Mv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tl.append(loss.item())

        # Validate
        model.eval()
        vl = []
        with torch.no_grad():
            for X, Y, M, Mv in val_dl:
                X, Y, M, Mv = (X.to(DEVICE), Y.to(DEVICE),
                                M.to(DEVICE), Mv.to(DEVICE))
                pred = model(X)
                loss = masked_mse_loss(pred, Y, M, Mv)
                vl.append(loss.item())

        avg_t, avg_v = np.mean(tl), np.mean(vl)
        elapsed = time.time() - t0
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        history['lr'].append(lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
                  f"Train: {avg_t:.6f} | Val: {avg_v:.6f} | "
                  f"LR: {lr:.2e} | {elapsed:.1f}s")

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val,
                'config': {
                    'L': L, 'h': h,
                    'patch_len': p_len, 'stride': s,
                    'd_model': D_MODEL, 'n_heads': N_HEADS,
                    'n_layers': N_LAYERS, 'd_ff': D_FF,
                    'dropout': DROPOUT, 'n_assets': n_assets,
                    'version': 'v2_scalar',
                }
            }, MODEL_DIR / f'{config}_best.pt')
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(best val: {best_val:.6f})")
                break

    with open(MODEL_DIR / f'{config}_history.json', 'w') as f:
        json.dump(history, f)

    print(f"  Best val loss: {best_val:.6f}")
    return best_val


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(f"TRAINING 9 PATCHTST MODELS")
    print(f"Lookbacks: {LOOKBACKS}, Horizons: {HORIZONS}")
    print(f"d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, "
          f"d_ff={D_FF}, dropout={DROPOUT}")
    print("=" * 60)

    results = {}
    for L in LOOKBACKS:
        for h in HORIZONS:
            results[(L, h)] = train_model(L, h)
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — VALIDATION LOSS SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(
        [[results[(L, h)] for h in HORIZONS] for L in LOOKBACKS],
        index=[f'L={L}' for L in LOOKBACKS],
        columns=[f'h={h}' for h in HORIZONS]
    )
    print(summary.round(6))
    summary.to_csv(MODEL_DIR / 'validation_loss_summary.csv')
    print(f"\nAll models saved to ./{MODEL_DIR}/")
