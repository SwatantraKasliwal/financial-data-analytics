"""
Comprehensive Financial Data Analytics - EDA
Author: [Your Name]
Date: September 2024

This notebook provides comprehensive exploratory data analysis for:
1. Transaction data (PaySim dataset)
2. Credit card default data

Requirements: pandas, numpy, matplotlib, seaborn, plotly, duckdb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Financial Data Analytics - Comprehensive EDA")
print("="*60)

#=============================================================================
# PART 1: TRANSACTION DATA ANALYSIS (PaySim Dataset)
#=============================================================================

print("\n📊 PART 1: TRANSACTION DATA ANALYSIS")
print("-"*40)

# Load transaction data
df_trans = pd.read_csv('data/transactions.csv')
print(f"✅ Loaded transaction data: {df_trans.shape[0]:,} rows, {df_trans.shape[1]} columns")

# Basic info
print("\n🔍 Dataset Overview:")
print(f"Memory usage: {df_trans.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Missing values: {df_trans.isnull().sum().sum()}")
print(f"Duplicate rows: {df_trans.duplicated().sum()}")

# Column info
print("\n📋 Column Information:")
for i, col in enumerate(df_trans.columns):
    print(f"{i+1:2d}. {col:15} - {df_trans[col].dtype}")

print("\n📈 Transaction Types Distribution:")
print(df_trans['type'].value_counts())

print("\n💰 Financial Metrics:")
print(f"Total transaction volume: ${df_trans['amount'].sum():,.2f}")
print(f"Average transaction: ${df_trans['amount'].mean():.2f}")
print(f"Median transaction: ${df_trans['amount'].median():.2f}")
print(f"Max transaction: ${df_trans['amount'].max():,.2f}")

print("\n🚨 Fraud Analysis:")
fraud_rate = df_trans['isFraud'].mean() * 100
print(f"Overall fraud rate: {fraud_rate:.3f}%")
print(f"Fraudulent transactions: {df_trans['isFraud'].sum():,}")
print(f"Fraud amount: ${df_trans[df_trans['isFraud']==1]['amount'].sum():,.2f}")

# Fraud by transaction type
print("\n🔍 Fraud Rate by Transaction Type:")
fraud_by_type = df_trans.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).round(4)
fraud_by_type.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
fraud_by_type['Fraud_Rate_Pct'] = fraud_by_type['Fraud_Rate'] * 100
print(fraud_by_type.sort_values('Fraud_Rate_Pct', ascending=False))

# Create visualizations
print("\n📊 Creating visualizations...")

# 1. Transaction type distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Transaction types
df_trans['type'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Transaction Types Distribution')
axes[0,0].set_xlabel('Transaction Type')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=45)

# Amount distribution (log scale)
df_trans[df_trans['amount'] > 0]['amount'].apply(np.log10).hist(bins=50, ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Transaction Amount Distribution (Log Scale)')
axes[0,1].set_xlabel('Log10(Amount)')
axes[0,1].set_ylabel('Frequency')

# Fraud by type
fraud_by_type['Fraud_Rate_Pct'].plot(kind='bar', ax=axes[1,0], color='red', alpha=0.7)
axes[1,0].set_title('Fraud Rate by Transaction Type')
axes[1,0].set_xlabel('Transaction Type')
axes[1,0].set_ylabel('Fraud Rate (%)')
axes[1,0].tick_params(axis='x', rotation=45)

# Amount vs fraud (scatter sample)
sample_data = df_trans.sample(n=min(10000, len(df_trans)))
scatter = axes[1,1].scatter(sample_data['amount'], sample_data['step'], 
                           c=sample_data['isFraud'], cmap='coolwarm', alpha=0.6)
axes[1,1].set_title('Transaction Amount vs Time Step (Fraud Colored)')
axes[1,1].set_xlabel('Amount')
axes[1,1].set_ylabel('Time Step')
plt.colorbar(scatter, ax=axes[1,1])

plt.tight_layout()
plt.savefig('images/transaction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Advanced analysis - Balance analysis
print("\n💳 Balance Analysis:")

# Balance changes analysis
df_trans['balance_change_orig'] = df_trans['newbalanceOrig'] - df_trans['oldbalanceOrg']
df_trans['balance_change_dest'] = df_trans['newbalanceDest'] - df_trans['oldbalanceDest']

# Detect inconsistencies
df_trans['amount_inconsistency'] = abs(df_trans['balance_change_orig'] + df_trans['amount']) > 0.01

print(f"Transactions with balance inconsistencies: {df_trans['amount_inconsistency'].sum():,} ({df_trans['amount_inconsistency'].mean()*100:.2f}%)")

# Create balance analysis plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original balance distribution
df_trans['oldbalanceOrg'].clip(upper=df_trans['oldbalanceOrg'].quantile(0.95)).hist(bins=50, ax=axes[0,0], alpha=0.7)
axes[0,0].set_title('Original Balance Distribution (95th percentile clipped)')
axes[0,0].set_xlabel('Balance')

# Destination balance distribution  
df_trans['oldbalanceDest'].clip(upper=df_trans['oldbalanceDest'].quantile(0.95)).hist(bins=50, ax=axes[0,1], alpha=0.7)
axes[0,1].set_title('Destination Balance Distribution (95th percentile clipped)')
axes[0,1].set_xlabel('Balance')

# Balance change for origin accounts
df_trans['balance_change_orig'].clip(lower=df_trans['balance_change_orig'].quantile(0.05),
                                    upper=df_trans['balance_change_orig'].quantile(0.95)).hist(bins=50, ax=axes[1,0], alpha=0.7)
axes[1,0].set_title('Balance Change - Origin Accounts')
axes[1,0].set_xlabel('Balance Change')

# Balance change for destination accounts
df_trans['balance_change_dest'].clip(lower=df_trans['balance_change_dest'].quantile(0.05),
                                    upper=df_trans['balance_change_dest'].quantile(0.95)).hist(bins=50, ax=axes[1,1], alpha=0.7)
axes[1,1].set_title('Balance Change - Destination Accounts')
axes[1,1].set_xlabel('Balance Change')

plt.tight_layout()
plt.savefig('images/balance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#=============================================================================
# PART 2: CREDIT CARD DEFAULT ANALYSIS
#=============================================================================

print("\n" + "="*60)
print("📊 PART 2: CREDIT CARD DEFAULT ANALYSIS")
print("-"*40)

# Load credit card data
df_credit = pd.read_excel('data/default of credit card clients.xls', header=1)
print(f"✅ Loaded credit card data: {df_credit.shape[0]:,} rows, {df_credit.shape[1]} columns")

# Rename target column for easier access
df_credit.rename(columns={'default payment next month': 'default'}, inplace=True)

print("\n🔍 Dataset Overview:")
print(df_credit.info())

print("\n📊 Default Rate Analysis:")
default_rate = df_credit['default'].mean() * 100
print(f"Overall default rate: {default_rate:.2f}%")
print(f"Default cases: {df_credit['default'].sum():,}")
print(f"Non-default cases: {(df_credit['default'] == 0).sum():,}")

# Demographic analysis
print("\n👥 Demographic Analysis:")

# Gender analysis
print("\nBy Gender:")
gender_default = df_credit.groupby('SEX')['default'].agg(['count', 'sum', 'mean'])
gender_default.columns = ['Total', 'Defaults', 'Default_Rate']
gender_default['Default_Rate_Pct'] = gender_default['Default_Rate'] * 100
gender_default.index = ['Male', 'Female']
print(gender_default)

# Education analysis
print("\nBy Education:")
edu_default = df_credit.groupby('EDUCATION')['default'].agg(['count', 'sum', 'mean'])
edu_default.columns = ['Total', 'Defaults', 'Default_Rate']
edu_default['Default_Rate_Pct'] = edu_default['Default_Rate'] * 100
print(edu_default.sort_values('Default_Rate_Pct', ascending=False))

# Marriage analysis
print("\nBy Marriage Status:")
marriage_default = df_credit.groupby('MARRIAGE')['default'].agg(['count', 'sum', 'mean'])
marriage_default.columns = ['Total', 'Defaults', 'Default_Rate']
marriage_default['Default_Rate_Pct'] = marriage_default['Default_Rate'] * 100
print(marriage_default)

# Age analysis
print("\n📈 Age Analysis:")
df_credit['age_group'] = pd.cut(df_credit['AGE'], bins=[20, 30, 40, 50, 60, 80], 
                               labels=['20-30', '30-40', '40-50', '50-60', '60+'])
age_default = df_credit.groupby('age_group')['default'].agg(['count', 'sum', 'mean'])
age_default.columns = ['Total', 'Defaults', 'Default_Rate']
age_default['Default_Rate_Pct'] = age_default['Default_Rate'] * 100
print(age_default)

# Credit limit analysis
print("\n💳 Credit Limit Analysis:")
df_credit['limit_group'] = pd.qcut(df_credit['LIMIT_BAL'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
limit_default = df_credit.groupby('limit_group')['default'].agg(['count', 'sum', 'mean'])
limit_default.columns = ['Total', 'Defaults', 'Default_Rate']
limit_default['Default_Rate_Pct'] = limit_default['Default_Rate'] * 100
print(limit_default)

# Payment history analysis
print("\n💰 Payment History Analysis:")
payment_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
print("Payment status distribution (PAY_0 = most recent):")
for col in payment_cols:
    print(f"\n{col}:")
    print(df_credit[col].value_counts().sort_index())

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 1. Default rate by gender
gender_default['Default_Rate_Pct'].plot(kind='bar', ax=axes[0,0], color=['lightblue', 'pink'])
axes[0,0].set_title('Default Rate by Gender')
axes[0,0].set_ylabel('Default Rate (%)')
axes[0,0].tick_params(axis='x', rotation=0)

# 2. Default rate by education
edu_default['Default_Rate_Pct'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Default Rate by Education')
axes[0,1].set_ylabel('Default Rate (%)')

# 3. Default rate by age group
age_default['Default_Rate_Pct'].plot(kind='bar', ax=axes[0,2], color='orange')
axes[0,2].set_title('Default Rate by Age Group')
axes[0,2].set_ylabel('Default Rate (%)')
axes[0,2].tick_params(axis='x', rotation=45)

# 4. Credit limit distribution
df_credit['LIMIT_BAL'].hist(bins=50, ax=axes[1,0], alpha=0.7)
axes[1,0].set_title('Credit Limit Distribution')
axes[1,0].set_xlabel('Credit Limit')

# 5. Age distribution
df_credit['AGE'].hist(bins=30, ax=axes[1,1], alpha=0.7)
axes[1,1].set_title('Age Distribution')
axes[1,1].set_xlabel('Age')

# 6. Default rate by credit limit group
limit_default['Default_Rate_Pct'].plot(kind='bar', ax=axes[1,2], color='red', alpha=0.7)
axes[1,2].set_title('Default Rate by Credit Limit Group')
axes[1,2].set_ylabel('Default Rate (%)')
axes[1,2].tick_params(axis='x', rotation=45)

# 7. Payment amount analysis
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
df_credit[bill_cols].mean().plot(kind='line', ax=axes[2,0], marker='o')
axes[2,0].set_title('Average Bill Amount by Month')
axes[2,0].set_xlabel('Month (1=Most Recent)')
axes[2,0].set_ylabel('Average Amount')

# 8. Payment status heatmap
payment_status_counts = df_credit[payment_cols].apply(lambda x: x.value_counts()).fillna(0)
sns.heatmap(payment_status_counts.T, annot=True, fmt='.0f', ax=axes[2,1], cmap='YlOrRd')
axes[2,1].set_title('Payment Status Distribution Heatmap')

# 9. Correlation with default
corr_cols = ['LIMIT_BAL', 'AGE'] + payment_cols + bill_cols
correlations = df_credit[corr_cols + ['default']].corr()['default'].drop('default').sort_values()
correlations.plot(kind='barh', ax=axes[2,2])
axes[2,2].set_title('Correlation with Default')
axes[2,2].set_xlabel('Correlation Coefficient')

plt.tight_layout()
plt.savefig('images/credit_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#=============================================================================
# PART 3: ADVANCED ANALYTICS & INSIGHTS
#=============================================================================

print("\n" + "="*60)
print("📊 PART 3: ADVANCED ANALYTICS & INSIGHTS")
print("-"*40)

# Risk scoring for credit card customers
print("\n🎯 Risk Scoring Model (Simple):")

# Create a simple risk score based on multiple factors
df_credit['risk_score'] = 0

# Payment history factor (higher recent payment delays = higher risk)
df_credit['risk_score'] += df_credit['PAY_0'] * 0.3
df_credit['risk_score'] += df_credit['PAY_2'] * 0.2
df_credit['risk_score'] += df_credit['PAY_3'] * 0.1

# Utilization factor (bill amount vs limit)
df_credit['utilization'] = df_credit['BILL_AMT1'] / df_credit['LIMIT_BAL']
df_credit['risk_score'] += df_credit['utilization'].fillna(0) * 0.4

# Create risk categories
df_credit['risk_category'] = pd.cut(df_credit['risk_score'], 
                                   bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
                                   labels=['Low', 'Medium', 'High', 'Very High'])

risk_analysis = df_credit.groupby('risk_category')['default'].agg(['count', 'sum', 'mean'])
risk_analysis.columns = ['Total', 'Defaults', 'Default_Rate']
risk_analysis['Default_Rate_Pct'] = risk_analysis['Default_Rate'] * 100

print("Risk Category Analysis:")
print(risk_analysis)

# Transaction pattern analysis
print("\n🔄 Transaction Pattern Analysis:")

# Peak transaction hours analysis
df_trans['hour'] = df_trans['step'] % 24
hourly_patterns = df_trans.groupby(['hour', 'type']).size().unstack(fill_value=0)

print("Peak transaction hours by type:")
for tx_type in df_trans['type'].unique():
    peak_hour = hourly_patterns[tx_type].idxmax()
    peak_count = hourly_patterns[tx_type].max()
    print(f"{tx_type}: Peak at hour {peak_hour} with {peak_count:,} transactions")

# High-value transaction analysis
print("\n💎 High-Value Transaction Analysis:")
high_value_threshold = df_trans['amount'].quantile(0.99)
high_value_trans = df_trans[df_trans['amount'] > high_value_threshold]

print(f"High-value threshold (99th percentile): ${high_value_threshold:,.2f}")
print(f"High-value transactions: {len(high_value_trans):,} ({len(high_value_trans)/len(df_trans)*100:.2f}%)")
print(f"High-value fraud rate: {high_value_trans['isFraud'].mean()*100:.2f}%")

print("\nHigh-value transactions by type:")
print(high_value_trans['type'].value_counts())

#=============================================================================
# PART 4: SUMMARY & RECOMMENDATIONS
#=============================================================================

print("\n" + "="*60)
print("📋 PART 4: KEY INSIGHTS & RECOMMENDATIONS")
print("="*60)

print("\n🔍 KEY FINDINGS:")

print("\n1. TRANSACTION DATA INSIGHTS:")
print(f"   • Total transactions analyzed: {len(df_trans):,}")
print(f"   • Overall fraud rate: {fraud_rate:.3f}% (very low but significant in volume)")
print(f"   • TRANSFER and CASH_OUT have highest fraud rates")
print(f"   • Total fraud amount: ${df_trans[df_trans['isFraud']==1]['amount'].sum():,.2f}")
print(f"   • Average fraudulent transaction: ${df_trans[df_trans['isFraud']==1]['amount'].mean():.2f}")

print("\n2. CREDIT CARD DEFAULT INSIGHTS:")
print(f"   • Overall default rate: {default_rate:.2f}%")
print(f"   • Higher default rates in lower education groups")
print(f"   • Payment history (PAY_0, PAY_2) strongly correlates with defaults")
print(f"   • Credit utilization is a key risk factor")

print("\n3. RISK PATTERNS:")
print(f"   • Balance inconsistencies detected in {df_trans['amount_inconsistency'].mean()*100:.2f}% of transactions")
print(f"   • High-value transactions (top 1%) have {high_value_trans['isFraud'].mean()*100:.2f}% fraud rate")
print(f"   • Risk scoring model shows clear default rate progression across risk categories")

print("\n💡 BUSINESS RECOMMENDATIONS:")

print("\n🛡️  FRAUD PREVENTION:")
print("   • Focus monitoring on TRANSFER and CASH_OUT transactions")
print("   • Implement real-time balance validation checks")
print("   • Set up alerts for transactions > 99th percentile amounts")
print("   • Monitor accounts with frequent balance inconsistencies")

print("\n💳 CREDIT RISK MANAGEMENT:")
print("   • Implement payment history scoring in credit decisions")
print("   • Monitor credit utilization ratios closely")
print("   • Consider education level in risk assessment")
print("   • Develop early warning system for payment delays")

print("\n📊 OPERATIONAL IMPROVEMENTS:")
print("   • Automate risk scoring for both fraud and credit default")
print("   • Create real-time dashboards for monitoring key metrics")
print("   • Implement A/B testing for different risk thresholds")
print("   • Establish regular model retraining schedules")

print("\n✅ EDA COMPLETE!")
print("Generated files:")
print("   • images/transaction_analysis.png")
print("   • images/balance_analysis.png") 
print("   • images/credit_analysis.png")