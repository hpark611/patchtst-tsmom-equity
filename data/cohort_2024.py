"""
Construction of the 2024 Equity Cohort
Selects 50 equities (20 Mega-Cap, 20 Large-Cap, 10 Mid-Cap) from CRSP
based on market capitalisation as of December 2024, with deduplication
against the 1990 cohort.
Requires: WRDS account, wrds Python package, cohort_1990_stocks.csv
"""

import wrds
import pandas as pd
import numpy as np
import os

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "YOUR_WRDS_USERNAME")

db = wrds.Connection(wrds_username=WRDS_USERNAME)

# Load 1990 cohort PERMNOs for deduplication
cohort_1990 = pd.read_csv('cohort_1990_stocks.csv')
permnos_1990 = set(cohort_1990['PERMNO'].astype(int))
print(f"Loaded {len(permnos_1990)} PERMNOs from 1990 cohort for dedup")

query = """
WITH dec2024 AS (
    SELECT a.permno,
           a.date,
           ABS(a.prc) AS price,
           a.shrout,
           ABS(a.prc) * a.shrout / 1000.0 AS mktcap_millions,
           b.shrcd,
           b.exchcd,
           b.siccd,
           b.comnam,
           b.ticker
    FROM crsp.dsf AS a
    INNER JOIN crsp.dsenames AS b
        ON a.permno = b.permno
        AND a.date BETWEEN b.namedt AND b.nameendt
    WHERE a.date BETWEEN '2024-12-01' AND '2024-12-31'
      AND b.shrcd IN (10, 11)
      AND b.exchcd IN (1, 2, 3)
      AND (b.siccd < 6000 OR b.siccd > 6999)
      AND a.prc IS NOT NULL
      AND a.shrout IS NOT NULL
),
last_day AS (
    SELECT permno, MAX(date) AS last_date
    FROM dec2024
    GROUP BY permno
),
selection AS (
    SELECT j.*
    FROM dec2024 j
    INNER JOIN last_day ld
        ON j.permno = ld.permno AND j.date = ld.last_date
),
data_range AS (
    SELECT permno,
           MIN(date) AS first_date,
           MAX(date) AS last_date,
           COUNT(*) AS n_obs
    FROM crsp.dsf
    WHERE permno IN (SELECT permno FROM selection)
    GROUP BY permno
)
SELECT s.permno,
       s.comnam AS company_name,
       s.ticker,
       s.siccd AS sic_code,
       s.exchcd AS exchange_code,
       s.shrcd AS share_code,
       s.date AS selection_date,
       s.price,
       s.shrout AS shares_outstanding_thousands,
       s.mktcap_millions,
       dr.first_date AS first_trading_date,
       dr.last_date AS last_trading_date,
       dr.n_obs AS total_observations
FROM selection s
INNER JOIN data_range dr
    ON s.permno = dr.permno
WHERE dr.n_obs >= 2520
ORDER BY s.mktcap_millions DESC;
"""

print("Querying WRDS for 2024 cohort candidates...")
df = db.raw_sql(query)
print(f"Total eligible stocks: {len(df)}")


def sic_to_sector(sic):
    sic = int(sic)
    if 100 <= sic <= 999: return "Agriculture/Mining"
    elif 1000 <= sic <= 1499: return "Mining"
    elif 1500 <= sic <= 1799: return "Construction"
    elif 2000 <= sic <= 3999: return "Manufacturing"
    elif 4000 <= sic <= 4999: return "Transportation/Utilities"
    elif 5000 <= sic <= 5199: return "Wholesale Trade"
    elif 5200 <= sic <= 5999: return "Retail Trade"
    elif 7000 <= sic <= 8999: return "Services"
    elif 9000 <= sic <= 9999: return "Public Administration"
    else: return "Other"


df['sector'] = df['sic_code'].apply(sic_to_sector)
df['exchange'] = df['exchange_code'].map({1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'})
df = df.sort_values('mktcap_millions', ascending=False).reset_index(drop=True)
df['in_1990_cohort'] = df['permno'].astype(int).isin(permnos_1990)


def select_tier(pool, n, tier_name, exclude_permnos):
    """Select top n stocks from pool, skipping any in exclude set."""
    selected = []
    skipped = []
    for _, row in pool.iterrows():
        if len(selected) >= n:
            break
        if int(row['permno']) in exclude_permnos:
            skipped.append(row)
            continue
        selected.append(row)
    sel_df = pd.DataFrame(selected)
    sel_df['tier'] = tier_name
    if skipped:
        skip_df = pd.DataFrame(skipped)
        print(f"\n  [{tier_name}] Skipped {len(skipped)} 1990-cohort overlaps:")
        for _, s in skip_df.iterrows():
            print(f"    - {s['ticker']:6s} {s['company_name'][:35]:35s} "
                  f"${s['mktcap_millions']:>12,.0f}M")
    return sel_df


print("\n" + "=" * 70)
print("TIER SELECTION WITH DEDUP")
print("=" * 70)

mega = select_tier(df, 20, 'Mega-Cap', permnos_1990)
mega_permnos = set(mega['permno'].astype(int))

large_pool = df[df['mktcap_millions'] <= 100000].copy()
large = select_tier(large_pool, 20, 'Large-Cap', permnos_1990 | mega_permnos)
large_permnos = set(large['permno'].astype(int))

mid_pool = df[df['mktcap_millions'] <= 20000].copy()
mid = select_tier(mid_pool, 10, 'Mid-Cap',
                  permnos_1990 | mega_permnos | large_permnos)

cohort = pd.concat([mega, large, mid], ignore_index=True)
cohort['rank'] = range(1, len(cohort) + 1)

cohort['years_of_data'] = (
    (pd.to_datetime(cohort['last_trading_date']) -
     pd.to_datetime(cohort['first_trading_date'])).dt.days / 365.25
).round(1)

cohort['status'] = np.where(
    pd.to_datetime(cohort['last_trading_date']) < pd.Timestamp('2024-12-01'),
    'Delisted/Acquired', 'Active'
)

output_cols = [
    'rank', 'permno', 'ticker', 'company_name', 'tier',
    'sector', 'sic_code', 'exchange',
    'mktcap_millions', 'price', 'shares_outstanding_thousands',
    'selection_date', 'first_trading_date', 'last_trading_date',
    'years_of_data', 'total_observations', 'status'
]
output = cohort[output_cols].copy()
output.columns = [
    'Rank', 'PERMNO', 'Ticker', 'Company Name', 'Tier',
    'Sector', 'SIC Code', 'Exchange',
    'Market Cap ($M)', 'Price ($)', 'Shares Out (000s)',
    'Selection Date', 'First Trading Date', 'Last Trading Date',
    'Years of Data', 'Total Obs', 'Status'
]
output['Market Cap ($M)'] = output['Market Cap ($M)'].round(1)
output['Price ($)'] = output['Price ($)'].round(2)

print("\n" + "=" * 70)
print("2024 COHORT — 50 STOCKS (20 MEGA + 20 LARGE + 10 MID)")
print("=" * 70)
for tier in ['Mega-Cap', 'Large-Cap', 'Mid-Cap']:
    subset = output[output['Tier'] == tier]
    print(f"\n--- {tier} ({len(subset)} stocks) ---")
    print(f"  Mkt cap range: ${subset['Market Cap ($M)'].min():,.0f}M – "
          f"${subset['Market Cap ($M)'].max():,.0f}M")
    for _, row in subset.iterrows():
        print(f"  {row['Rank']:3d}. {row['Ticker']:6s} "
              f"{row['Company Name'][:35]:35s} "
              f"${row['Market Cap ($M)']:>12,.0f}M "
              f"{row['Sector'][:20]}")

output.to_csv('cohort_2024_stocks.csv', index=False)
print(f"\nSaved cohort_2024_stocks.csv ({len(output)} stocks)")

db.close()
