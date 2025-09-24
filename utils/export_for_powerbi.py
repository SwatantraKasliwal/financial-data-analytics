"""
Power BI Data Export Utility

This script exports processed financial data in Power BI compatible formats.
Exports cleaned datasets optimized for Power BI visualization and analysis.

Author: Financial Analytics Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from kaggle_data_fetcher import KaggleDataFetcher
except ImportError:
    print("Warning: Could not import KaggleDataFetcher. Using sample data.")
    KaggleDataFetcher = None

class PowerBIDataExporter:
    """Export financial data optimized for Power BI analysis"""
    
    def __init__(self, output_dir='powerbi/data'):
        self.output_dir = output_dir
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✅ Created output directory: {self.output_dir}")
    
    def fetch_and_process_data(self):
        """Fetch and process all required datasets"""
        print("🔄 Fetching and processing datasets...")
        
        # Initialize data fetcher
        if KaggleDataFetcher:
            fetcher = KaggleDataFetcher()
        else:
            fetcher = None
        
        # Get transaction data
        try:
            if fetcher:
                transactions_df = fetcher.get_paysim_data()
                print("✅ PaySim transaction data fetched from Kaggle")
            else:
                transactions_df = self.generate_sample_transactions()
                print("✅ Sample transaction data generated")
        except Exception as e:
            print(f"⚠️ Error fetching transaction data: {e}")
            transactions_df = self.generate_sample_transactions()
            print("✅ Fallback to sample transaction data")
        
        # Get credit card data
        try:
            if fetcher:
                credit_df = fetcher.get_uci_credit_data()
                print("✅ UCI credit card data fetched")
            else:
                credit_df = self.generate_sample_credit()
                print("✅ Sample credit data generated")
        except Exception as e:
            print(f"⚠️ Error fetching credit data: {e}")
            credit_df = self.generate_sample_credit()
            print("✅ Fallback to sample credit data")
        
        return transactions_df, credit_df
        
        return transactions_df, credit_df
    
    def generate_sample_transactions(self, n_samples=10000):
        """Generate sample transaction data for Power BI"""
        np.random.seed(42)
        
        # Transaction types
        transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        
        # Generate date range (last 12 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        transactions = []
        for i in range(n_samples):
            transaction_date = start_date + timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            transaction_type = np.random.choice(transaction_types)
            
            # Amount based on transaction type
            if transaction_type == 'CASH_OUT':
                amount = np.random.lognormal(7, 1.5)  # Higher amounts
            elif transaction_type == 'TRANSFER':
                amount = np.random.lognormal(6, 1.2)
            else:
                amount = np.random.lognormal(5, 1)
            
            # Fraud probability (2% overall)
            is_fraud = 1 if np.random.random() < 0.02 else 0
            
            # If fraud, typically higher amounts
            if is_fraud:
                amount *= np.random.uniform(2, 5)
            
            transactions.append({
                'transaction_id': f'T{i+1:06d}',
                'customer_id': f'C{np.random.randint(1, 5000):05d}',
                'transaction_date': transaction_date,
                'transaction_type': transaction_type,
                'amount': round(amount, 2),
                'is_fraud': is_fraud,
                'old_balance': round(np.random.uniform(0, 50000), 2),
                'new_balance': round(np.random.uniform(0, 50000), 2),
                'hour_of_day': transaction_date.hour,
                'day_of_week': transaction_date.strftime('%A'),
                'month_year': transaction_date.strftime('%Y-%m')
            })
        
        return pd.DataFrame(transactions)
    
    def generate_sample_credit(self, n_samples=5000):
        """Generate sample credit card data for Power BI"""
        np.random.seed(42)
        
        credit_data = []
        for i in range(n_samples):
            # Basic demographics
            age = np.random.randint(21, 70)
            education = np.random.choice([1, 2, 3, 4, 5, 6])  # Education levels
            marriage = np.random.choice([1, 2, 3])  # Marital status
            
            # Financial data
            limit_bal = np.random.choice([50000, 100000, 200000, 300000, 500000])
            
            # Payment history (last 6 months)
            pay_history = [np.random.choice([-1, 0, 1, 2, 3]) for _ in range(6)]
            
            # Bill amounts and payments
            bill_amounts = [np.random.uniform(0, limit_bal * 0.8) for _ in range(6)]
            pay_amounts = [max(0, bill + np.random.normal(0, bill * 0.3)) for bill in bill_amounts]
            
            # Calculate risk factors
            total_bill = sum(bill_amounts)
            total_payments = sum(pay_amounts)
            utilization_rate = total_bill / limit_bal if limit_bal > 0 else 0
            
            # Default probability based on risk factors
            risk_score = (
                (age < 25) * 0.1 +
                (utilization_rate > 0.8) * 0.3 +
                (sum(pay_history) > 3) * 0.4 +
                (education <= 2) * 0.1
            )
            
            default_probability = min(0.95, max(0.05, risk_score + np.random.normal(0, 0.1)))
            default_next_month = 1 if default_probability > 0.5 else 0
            
            credit_data.append({
                'customer_id': f'C{i+1:05d}',
                'limit_bal': limit_bal,
                'age': age,
                'education': education,
                'marriage': marriage,
                'pay_0': pay_history[0],
                'pay_2': pay_history[1],
                'pay_3': pay_history[2],
                'pay_4': pay_history[3],
                'pay_5': pay_history[4],
                'pay_6': pay_history[5],
                'bill_amt1': round(bill_amounts[0], 2),
                'bill_amt2': round(bill_amounts[1], 2),
                'bill_amt3': round(bill_amounts[2], 2),
                'bill_amt4': round(bill_amounts[3], 2),
                'bill_amt5': round(bill_amounts[4], 2),
                'bill_amt6': round(bill_amounts[5], 2),
                'pay_amt1': round(pay_amounts[0], 2),
                'pay_amt2': round(pay_amounts[1], 2),
                'pay_amt3': round(pay_amounts[2], 2),
                'pay_amt4': round(pay_amounts[3], 2),
                'pay_amt5': round(pay_amounts[4], 2),
                'pay_amt6': round(pay_amounts[5], 2),
                'utilization_rate': round(utilization_rate, 3),
                'default_probability': round(default_probability, 3),
                'default_next_month': default_next_month,
                'risk_category': 'High Risk' if default_probability > 0.7 else 'Medium Risk' if default_probability > 0.3 else 'Low Risk'
            })
        
        return pd.DataFrame(credit_data)
    
    def process_transactions_for_powerbi(self, df):
        """Process transaction data for Power BI optimization"""
        print("🔄 Processing transaction data for Power BI...")
        
        # Ensure proper data types
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Create additional calculated columns for Power BI
        df['year'] = df['transaction_date'].dt.year
        df['month'] = df['transaction_date'].dt.month
        df['quarter'] = df['transaction_date'].dt.quarter
        df['weekday'] = df['transaction_date'].dt.day_name()
        df['is_weekend'] = df['transaction_date'].dt.weekday >= 5
        
        # Amount categories
        amount_bins = [0, 100, 1000, 10000, float('inf')]
        amount_labels = ['Small', 'Medium', 'Large', 'Very Large']
        df['amount_category'] = pd.cut(df['amount'], bins=amount_bins, labels=amount_labels)
        
        # Convert categorical to string to avoid export issues
        df['amount_category'] = df['amount_category'].astype(str)
        
        # Time-based features
        hour_bins = [0, 6, 12, 18, 24]
        hour_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        df['hour_category'] = pd.cut(df['hour_of_day'], bins=hour_bins, labels=hour_labels)
        df['hour_category'] = df['hour_category'].astype(str)
        
        # Fraud indicators
        df['high_amount_flag'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['unusual_time_flag'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
        
        return df
    
    def process_credit_for_powerbi(self, df):
        """Process credit data for Power BI optimization"""
        print("🔄 Processing credit data for Power BI...")
        
        # Create age groups
        age_bins = [0, 25, 35, 45, 55, 100]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        df['age_group'] = df['age_group'].astype(str)
        
        # Education categories
        education_map = {1: 'Graduate', 2: 'University', 3: 'High School', 
                        4: 'Middle School', 5: 'Elementary', 6: 'Other'}
        df['education_level'] = df['education'].map(education_map)
        
        # Marriage status
        marriage_map = {1: 'Married', 2: 'Single', 3: 'Other'}
        df['marital_status'] = df['marriage'].map(marriage_map)
        
        # Payment behavior analysis
        payment_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
        df['avg_payment_delay'] = df[payment_cols].mean(axis=1)
        df['max_payment_delay'] = df[payment_cols].max(axis=1)
        df['payment_consistency'] = df[payment_cols].std(axis=1)
        
        # Bill amount analysis
        bill_cols = ['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']
        df['avg_bill_amount'] = df[bill_cols].mean(axis=1)
        df['total_bill_amount'] = df[bill_cols].sum(axis=1)
        
        # Payment amount analysis
        pay_cols = ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
        df['avg_payment_amount'] = df[pay_cols].mean(axis=1)
        df['total_payment_amount'] = df[pay_cols].sum(axis=1)
        
        # Credit utilization
        df['current_balance'] = df['bill_amt1']  # Most recent bill
        df['utilization_rate'] = np.where(df['limit_bal'] > 0, df['current_balance'] / df['limit_bal'], 0)
        
        # Utilization categories with string conversion
        util_bins = [0, 0.3, 0.7, 1.0, float('inf')]
        util_labels = ['Low', 'Medium', 'High', 'Over Limit']
        df['utilization_category'] = pd.cut(df['utilization_rate'], bins=util_bins, labels=util_labels)
        df['utilization_category'] = df['utilization_category'].astype(str)
        
        return df
    
    def create_summary_tables(self, transactions_df, credit_df):
        """Create summary tables for Power BI dashboards"""
        print("🔄 Creating summary tables...")
        
        # Monthly transaction summary
        monthly_summary = transactions_df.groupby(['year', 'month', 'transaction_type']).agg({
            'amount': ['sum', 'mean', 'count'],
            'is_fraud': 'sum'
        }).round(2)
        monthly_summary.columns = ['total_amount', 'avg_amount', 'transaction_count', 'fraud_count']
        monthly_summary = monthly_summary.reset_index()
        monthly_summary['fraud_rate'] = (monthly_summary['fraud_count'] / monthly_summary['transaction_count'] * 100).round(2)
        
        # Customer segmentation (RFM-like analysis)
        customer_summary = transactions_df.groupby('customer_id').agg({
            'transaction_date': ['max', 'count'],
            'amount': ['sum', 'mean'],
            'is_fraud': 'sum'
        }).round(2)
        customer_summary.columns = ['last_transaction', 'frequency', 'total_spent', 'avg_transaction', 'fraud_incidents']
        customer_summary = customer_summary.reset_index()
        
        # Calculate recency (days since last transaction)
        customer_summary['recency_days'] = (pd.Timestamp.now() - customer_summary['last_transaction']).dt.days
        
        # Customer segments with string conversion
        customer_summary['value_segment'] = pd.qcut(customer_summary['total_spent'], 3, labels=['Low', 'Medium', 'High'])
        customer_summary['value_segment'] = customer_summary['value_segment'].astype(str)
        
        customer_summary['frequency_segment'] = pd.qcut(customer_summary['frequency'], 3, labels=['Infrequent', 'Regular', 'Frequent'])
        customer_summary['frequency_segment'] = customer_summary['frequency_segment'].astype(str)
        
        # Fraud analysis summary
        fraud_summary = transactions_df[transactions_df['is_fraud'] == 1].groupby(['transaction_type', 'amount_category']).agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        fraud_summary.columns = ['fraud_total_amount', 'fraud_avg_amount', 'fraud_count', 'affected_customers']
        fraud_summary = fraud_summary.reset_index()
        
        # Credit risk summary
        credit_summary = credit_df.groupby(['risk_category', 'age_group']).agg({
            'limit_bal': ['mean', 'sum'],
            'utilization_rate': 'mean',
            'default_probability': 'mean',
            'customer_id': 'count'
        }).round(3)
        credit_summary.columns = ['avg_credit_limit', 'total_credit_limit', 'avg_utilization', 'avg_default_prob', 'customer_count']
        credit_summary = credit_summary.reset_index()
        
        return {
            'monthly_summary': monthly_summary,
            'customer_segments': customer_summary,
            'fraud_analysis': fraud_summary,
            'credit_risk_summary': credit_summary
        }
    
    def export_to_csv(self, dataframes_dict):
        """Export all dataframes to CSV files for Power BI"""
        print("💾 Exporting data to CSV files...")
        
        exported_files = []
        for name, df in dataframes_dict.items():
            filename = f"{name}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Clean data for CSV export
            df_clean = df.copy()
            
            # Handle datetime columns
            for col in df_clean.columns:
                if df_clean[col].dtype == 'datetime64[ns]':
                    df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Replace inf and NaN values
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            df_clean = df_clean.fillna('')
            
            # Export to CSV
            df_clean.to_csv(filepath, index=False, encoding='utf-8')
            exported_files.append(filepath)
            print(f"✅ Exported {name}: {len(df)} rows → {filepath}")
        
        return exported_files
    
    def create_data_dictionary(self, dataframes_dict):
        """Create data dictionary for Power BI users"""
        print("📝 Creating data dictionary...")
        
        data_dict = []
        
        column_descriptions = {
            # Transaction data
            'transaction_id': 'Unique identifier for each transaction',
            'customer_id': 'Unique identifier for each customer',
            'transaction_date': 'Date and time of transaction',
            'transaction_type': 'Type of transaction (PAYMENT, TRANSFER, etc.)',
            'amount': 'Transaction amount in currency units',
            'is_fraud': 'Fraud indicator (1=fraud, 0=legitimate)',
            'amount_category': 'Transaction amount category (Small, Medium, Large, Very Large)',
            'hour_category': 'Time of day category (Night, Morning, Afternoon, Evening)',
            'weekday': 'Day of the week',
            'is_weekend': 'Weekend indicator (True/False)',
            
            # Credit data
            'limit_bal': 'Credit limit amount',
            'age': 'Customer age in years',
            'education': 'Education level (1-6 scale)',
            'marriage': 'Marital status (1=married, 2=single, 3=other)',
            'default_probability': 'Calculated probability of default (0-1)',
            'risk_category': 'Risk classification (Low, Medium, High)',
            'utilization_rate': 'Credit utilization ratio (balance/limit)',
            'age_group': 'Age range category',
            'education_level': 'Education level description',
            'marital_status': 'Marital status description',
            
            # Summary fields
            'fraud_rate': 'Percentage of fraudulent transactions',
            'total_amount': 'Sum of transaction amounts',
            'avg_amount': 'Average transaction amount',
            'transaction_count': 'Number of transactions',
            'recency_days': 'Days since last transaction',
            'frequency': 'Number of transactions per customer',
            'value_segment': 'Customer value classification',
        }
        
        for table_name, df in dataframes_dict.items():
            for column in df.columns:
                data_dict.append({
                    'Table': table_name,
                    'Column': column,
                    'Data_Type': str(df[column].dtype),
                    'Description': column_descriptions.get(column, 'No description available'),
                    'Sample_Values': str(df[column].dropna().head(3).tolist()) if len(df) > 0 else 'No data'
                })
        
        # Export data dictionary
        dict_df = pd.DataFrame(data_dict)
        dict_filepath = os.path.join(self.output_dir, 'data_dictionary.csv')
        dict_df.to_csv(dict_filepath, index=False)
        print(f"✅ Data dictionary exported: {dict_filepath}")
        
        return dict_filepath
    
    def run_export(self):
        """Main method to run the complete export process"""
        print("🚀 Starting Power BI data export process...")
        print("=" * 60)
        
        # Fetch and process data
        transactions_df, credit_df = self.fetch_and_process_data()
        
        # Process for Power BI
        transactions_processed = self.process_transactions_for_powerbi(transactions_df)
        credit_processed = self.process_credit_for_powerbi(credit_df)
        
        # Create summary tables
        summary_tables = self.create_summary_tables(transactions_processed, credit_processed)
        
        # Prepare all dataframes for export
        all_dataframes = {
            'transactions_processed': transactions_processed,
            'credit_data_processed': credit_processed,
            **summary_tables
        }
        
        # Export to CSV
        exported_files = self.export_to_csv(all_dataframes)
        
        # Create data dictionary
        dict_file = self.create_data_dictionary(all_dataframes)
        
        print("=" * 60)
        print("✅ Power BI export completed successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Files exported: {len(exported_files)} CSV files")
        print("📝 Data dictionary created")
        print("\n🎯 Next Steps:")
        print("1. Open Power BI Desktop")
        print("2. Import CSV files from the output directory")
        print("3. Follow the Power BI Dashboard Guide")
        print("4. Create your financial analytics dashboard!")
        
        return exported_files

def main():
    """Main execution function"""
    print("🎯 Power BI Data Export Utility")
    print("Preparing financial data for Power BI visualization")
    print("=" * 60)
    
    # Create exporter instance
    exporter = PowerBIDataExporter()
    
    # Run export process
    try:
        exported_files = exporter.run_export()
        return exported_files
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        print("Please check the error and try again.")
        return None

if __name__ == "__main__":
    main()