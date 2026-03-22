"""
Data Cleaning and Pre-Processing
Pulls daily returns from CRSP, computes log returns, applies 60-day EWMA
volatility scaling, and saves the preprocessed data as a pickle file.
Requires: WRDS account, cohort_1990_stocks.csv, cohort_2024_stocks.csv
"""

import wrds
import pandas as pd
import numpy as np
import pickle
import os

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "YOUR_WRDS_USERNAME")

db = wrds.Connection(wrds_username=WRDS_USERNAME)

# STEP 1: Load both cohorts and combine unique PERMNOs
cohort_1990 = pd.read_csv('cohort_1990_stocks.csv')
cohort_2024 = pd.read_csv('cohort_2024_stocks.csv')
all_permnos = list(set(cohort_1990['PERMNO'].tolist() + cohort_2024['PERMNO'].tolist()))
print(f"Total unique PERMNOs: {len(all_permnos)}")

# Build metadata lookup
meta = pd.concat([
    cohort_1990[['PERMNO', 'Ticker', 'Company Name', 'Tier']].assign(Cohort='1990'),
    cohort_2024[['PERMNO', 'Ticker', 'Company Name', 'Tier']].assign(Cohort='2024')
]).drop_duplicates(subset='PERMNO', keep='first').set_index('PERMNO')

# STEP 2: Pull daily returns from CRSP (Jan 1990 – Dec 2024)
permno_str = ', '.join(str(p) for p in all_permnos)
query = f"""
SELECT permno, date, ret
FROM crsp.dsf
WHERE permno IN ({permno_str})
  AND date BETWEEN '1990-01-02' AND '2024-12-31'
  AND ret IS NOT NULL
ORDER BY permno, date;
"""

print("Pulling daily returns from CRSP...")
raw = db.raw_sql(query)
raw['date'] = pd.to_datetime(raw['date'])
raw['permno'] = raw['permno'].astype(int)
print(f"Fetched {len(raw):,} daily return observations")
print(f"Date range: {raw['date'].min().date()} to {raw['date'].max().date()}")
print(f"Unique PERMNOs with data: {raw['permno'].nunique()}")
db.close()

# STEP 3: Compute log returns
raw['log_ret'] = np.log(1 + raw['ret'])

n_inf = np.isinf(raw['log_ret']).sum()
n_nan = raw['log_ret'].isna().sum()
print(f"\nLog return quality: {n_inf} inf, {n_nan} NaN")
if n_inf > 0 or n_nan > 0:
    print("  Dropping invalid log returns...")
    raw = raw[np.isfinite(raw['log_ret'])].copy()

# STEP 4: Pivot to wide format (date x permno)
log_ret_wide = raw.pivot_table(index='date', columns='permno', values='log_ret')
print(f"\nWide panel shape: {log_ret_wide.shape}")
print(f"  {log_ret_wide.shape[0]} trading days x {log_ret_wide.shape[1]} assets")

coverage = log_ret_wide.notna().sum() / len(log_ret_wide) * 100
print(f"\nPer-asset coverage (% of trading days with data):")
print(f"  Min: {coverage.min():.1f}%  Median: {coverage.median():.1f}%  "
      f"Max: {coverage.max():.1f}%")

# STEP 5: Volatility scaling (60-day EWMA, strictly ex-ante)
VOL_WINDOW = 60
print(f"\nComputing {VOL_WINDOW}-day EWMA volatility (ex-ante, shifted)...")
ewma_vol = log_ret_wide.shift(1).ewm(span=VOL_WINDOW, min_periods=30).std()

VOL_FLOOR = 1e-6
ewma_vol = ewma_vol.clip(lower=VOL_FLOOR)

# Volatility-scaled returns: x_{i,t} = r_{i,t} / sigma_hat_{i,t}
scaled_ret = log_ret_wide / ewma_vol

valid_start = scaled_ret.dropna(how='all').index[0]
scaled_ret = scaled_ret.loc[valid_start:]
log_ret_wide = log_ret_wide.loc[valid_start:]
ewma_vol = ewma_vol.loc[valid_start:]

print(f"After burn-in, panel starts: {scaled_ret.index[0].date()}")
print(f"Scaled returns shape: {scaled_ret.shape}")

asset_stds = scaled_ret.std()
print(f"\nScaled return std across assets:")
print(f"  Min: {asset_stds.min():.3f}  Median: {asset_stds.median():.3f}  "
      f"Max: {asset_stds.max():.3f}")

# STEP 6: Chronological train/val/test split
splits = {
    'train': ('1990-01-01', '2018-12-31'),
    'val':   ('2019-01-01', '2020-12-31'),
    'test':  ('2021-01-01', '2024-12-31'),
}

print("\nChronological splits:")
for name, (s, e) in splits.items():
    mask = (scaled_ret.index >= s) & (scaled_ret.index <= e)
    n = mask.sum()
    print(f"  {name:6s}: {s} to {e} — {n} trading days")

# STEP 7: Save preprocessed data
data = {
    'log_returns':    log_ret_wide,
    'scaled_returns': scaled_ret,
    'ewma_vol':       ewma_vol,
    'permnos':        list(log_ret_wide.columns),
    'dates':          log_ret_wide.index,
    'metadata':       meta,
    'splits':         splits,
    'params': {
        'vol_window': VOL_WINDOW,
        'vol_floor':  VOL_FLOOR,
    }
}

with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"\nSaved preprocessed_data.pkl")
print("\n" + "=" * 70)
print("READY FOR TENSOR CONSTRUCTION & PATCHTST TRAINING")
print("=" * 70)
