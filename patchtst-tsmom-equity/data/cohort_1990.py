"""
Construction of the 1990 Equity Cohort
Selects 50 equities (20 Mega-Cap, 20 Large-Cap, 10 Mid-Cap) from CRSP
based on market capitalisation as of January 1990.
Requires: WRDS account, wrds Python package
"""

import wrds
import pandas as pd
import numpy as np
import os

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "YOUR_WRDS_USERNAME")

db = wrds.Connection(wrds_username=WRDS_USERNAME)

query = """
WITH jan1990 AS (
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
    WHERE a.date BETWEEN '1990-01-02' AND '1990-01-31'
      AND b.shrcd IN (10, 11)
      AND b.exchcd IN (1, 2, 3)
      AND (b.siccd < 6000 OR b.siccd > 6999)
      AND a.prc IS NOT NULL
      AND a.shrout IS NOT NULL
),
last_day AS (
    SELECT permno, MAX(date) AS last_date
    FROM jan1990
    GROUP BY permno
),
selection AS (
    SELECT j.*
    FROM jan1990 j
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
WHERE dr.last_date >= '2000-01-01'
ORDER BY s.mktcap_millions DESC;
"""

print("Querying WRDS for 1990 cohort candidates...")
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

print(f"\nMarket cap distribution ($M):")
print(df['mktcap_millions'].describe().round(1))

# Tier selection
mega = df.head(20).copy()
mega['tier'] = 'Mega-Cap'

large_pool = df[df['mktcap_millions'] <= 6000].head(20).copy()
large_pool['tier'] = 'Large-Cap'

large_permnos = set(large_pool['permno'])
mid_pool = df[(df['mktcap_millions'] <= 1200) & (~df['permno'].isin(large_permnos))].head(10).copy()
mid_pool['tier'] = 'Mid-Cap'

cohort = pd.concat([mega, large_pool, mid_pool], ignore_index=True)
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
print("1990 COHORT — 50 STOCKS (20 MEGA + 20 LARGE + 10 MID)")
print("=" * 70)
for tier in ['Mega-Cap', 'Large-Cap', 'Mid-Cap']:
    subset = output[output['Tier'] == tier]
    print(f"\n--- {tier} ({len(subset)} stocks) ---")
    print(f"  Mkt cap range: ${subset['Market Cap ($M)'].min():,.0f}M – "
          f"${subset['Market Cap ($M)'].max():,.0f}M")
    for _, row in subset.iterrows():
        flag = " [DELISTED]" if row['Status'] == 'Delisted/Acquired' else ""
        print(f"  {row['Rank']:3d}. {row['Ticker']:6s} "
              f"{row['Company Name'][:35]:35s} "
              f"${row['Market Cap ($M)']:>10,.0f}M "
              f"{row['Sector'][:20]}{flag}")

output.to_csv('cohort_1990_stocks.csv', index=False)
print(f"\nSaved cohort_1990_stocks.csv ({len(output)} stocks)")

db.close()
