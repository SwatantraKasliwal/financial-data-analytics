"""
Transaction-level EDA (PaySim / Kaggle credit card transactions)
Save as: notebooks/01_transactions_eda.py or convert to .ipynb
Requirements: pip install pandas numpy matplotlib seaborn duckdb plotly
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from pathlib import Path

sns.set_theme(style="whitegrid")
pd.options.display.max_columns = 120

DATA = Path("C:/Users/swata/skill/projects/Data Analytics/Financial Data Analytics/financial-data-analytics/data/transactions.csv")  # adjust path as needed

# 1) Load
print("Loading:", DATA)
df = pd.read_csv(DATA)
print("Rows,Cols:", df.shape)
print(df.columns.tolist())
df.head()

# 2) Quick info & missing
print(df.info())
print("\nMissing per column:\n", df.isna().sum())

# 3) Standardize and detect key columns
cols = [c.lower().strip() for c in df.columns]
df.columns = cols

# normalize a few common names (examples)
if 'amount' not in df.columns:
    for alt in ['amt','transactionamount','amt_trans']:
        if alt in df.columns:
            df.rename(columns={alt:'amount'}, inplace=True)
            break

# 4) Convert time columns if present
if 'step' in df.columns:  # PaySim uses step (time step)
    # treat 'step' as integer timestep; create day/hour proxies
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['day'] = (df['step'] // 24) + 1
    df['hour'] = df['step'] % 24
if 'time' in df.columns:
    try:
        df['time'] = pd.to_datetime(df['time'], unit='s')  # if UNIX seconds
    except Exception:
        try: df['time'] = pd.to_datetime(df['time'])
        except: pass
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['date'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['dayofweek'] = df['time'].dt.day_name()

# 5) Basic numeric cleaning
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
df = df.drop_duplicates()
print("After dedup:", df.shape)

# 6) High-level KPIs
total_tx = len(df)
total_volume = df['amount'].sum()
unique_accounts = df[['nameorig','namedest']].stack().nunique() if {'nameorig','namedest'}.issubset(df.columns) else df['nameorig'].nunique() if 'nameorig' in df.columns else None
print(f"Total tx: {total_tx:,}, Total volume: {total_volume:,.2f}, Unique accounts (est): {unique_accounts}")

# 7) If isFraud/class exists, compute fraud rates
fraud_col = None
for c in ['isfraud','class','is_fraud','fraud']:
    if c in df.columns:
        fraud_col = c; break
if fraud_col:
    print("Using fraud column:", fraud_col)
    fraud_rate = df[fraud_col].mean()
    print("Fraud rate:", fraud_rate)
    # fraud by type
    if 'type' in df.columns:
        print(df.groupby('type')[fraud_col].mean().sort_values(ascending=False).head())

# 8) Time series trend (group by day or monthly if date available)
if 'date' in df.columns:
    ts = df.groupby('date').agg(tx_count=('amount','count'), tx_sum=('amount','sum')).reset_index()
    ts['tx_count'].plot(title="Transactions per day", figsize=(12,4))
    plt.show()
elif 'day' in df.columns:
    ts = df.groupby('day').agg(tx_count=('amount','count'), tx_sum=('amount','sum')).reset_index()
    ts.plot(x='day', y='tx_count', title='Transactions by step-day', figsize=(12,4))
    plt.show()

# 9) Distribution of amounts
plt.figure(figsize=(8,4))
sns.histplot(df[df['amount']>0]['amount'].clip(upper=df['amount'].quantile(0.99)), bins=100, kde=False)
plt.title("Transaction amount distribution (clipped 99th pct)")
plt.show()

# 10) Top merchants / top senders
if 'nameorig' in df.columns:
    top_senders = df.groupby('nameorig')['amount'].sum().sort_values(ascending=False).head(10)
    print("Top senders by amount:\n", top_senders)

if 'namedest' in df.columns:
    top_receivers = df.groupby('namedest')['amount'].sum().sort_values(ascending=False).head(10)
    print("Top receivers by amount:\n", top_receivers)

# 11) Type breakdown
if 'type' in df.columns:
    print("Transaction types:\n", df['type'].value_counts())

# 12) Hourly heatmap (if hour)
if 'hour' in df.columns and 'dayofweek' in df.columns:
    pivot = df.pivot_table(index='dayofweek', columns='hour', values='amount', aggfunc='count', fill_value=0)
    # reorder days
    days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot.reindex(days_order).fillna(0)
    plt.figure(figsize=(12,4)); sns.heatmap(pivot, cmap="Blues"); plt.title("Tx count heatmap (day vs hour)"); plt.show()

# 13) Save cleaned CSV for downstream (SQL / Power BI)
df.to_csv("data/transactions_cleaned.csv", index=False)
print("Saved cleaned data to data/transactions_cleaned.csv")
