"""
Kaggle Data Fetcher Utility for Financial Data Analytics
Handles downloading and caching data from Kaggle datasets with intelligent fallbacks
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import tempfile
import warnings
import kagglehub
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')

class KaggleDataFetcher:
    """Enhanced utility class to fetch financial data from multiple sources"""
    
    def __init__(self, cache_dir="data_cache"):
        """Initialize the data fetcher with caching capabilities"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Available datasets with enhanced configurations
        self.datasets = {
            "paysim_transactions": {
                "kagglehub_id": "ealaxi/paysim1",
                "description": "PaySim Mobile Money Simulator - Financial transaction data",
                "size": "470MB",
                "file_pattern": "*.csv"
            },
            "credit_card_fraud": {
                "kagglehub_id": "mlg-ulb/creditcardfraud",
                "description": "Credit Card Fraud Detection Dataset",
                "size": "150MB", 
                "file_pattern": "creditcard.csv"
            },
            "credit_risk": {
                "kagglehub_id": "laotse/credit-risk-dataset",
                "description": "Credit Risk Assessment Dataset",
                "size": "2MB",
                "file_pattern": "credit_risk_dataset.csv"
            },
            "loan_default": {
                "kagglehub_id": "yasserh/loan-default-dataset", 
                "description": "Loan Default Prediction Dataset",
                "size": "10MB",
                "file_pattern": "Loan_Default.csv"
            },
            "financial_distress": {
                "kagglehub_id": "shebrahimi/financial-distress",
                "description": "Financial Distress Prediction Dataset",
                "size": "5MB",
                "file_pattern": "Financial Distress.csv"
            }
        }
    
    def download_from_kagglehub(self, dataset_name):
        """Download dataset using kagglehub (preferred method)"""
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            dataset_info = self.datasets[dataset_name]
            cache_file = self.cache_dir / f"{dataset_name}.csv"
            
            # Check if already cached
            if cache_file.exists():
                print(f"✅ Using cached dataset: {cache_file}")
                return str(cache_file)
            
            print(f"📥 Downloading {dataset_name} from Kaggle...")
            
            # Download using kagglehub
            path = kagglehub.dataset_download(dataset_info["kagglehub_id"])
            
            # Find the CSV file in the downloaded path
            csv_files = list(Path(path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset")
            
            # Use the first CSV file found
            source_file = csv_files[0]
            
            # Copy to cache
            import shutil
            shutil.copy2(source_file, cache_file)
            
            print(f"✅ Successfully downloaded and cached: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            print(f"❌ Error downloading via kagglehub: {e}")
            return None
    
    def load_transaction_data(self, dataset_name="paysim_transactions", sample_size=None):
        """Load transaction data with intelligent fallbacks"""
        try:
            # Try kagglehub first
            file_path = self.download_from_kagglehub(dataset_name)
            
            if file_path and Path(file_path).exists():
                print(f"📊 Loading data from: {file_path}")
                df = pd.read_csv(file_path)
                
                # Standardize column names
                df.columns = df.columns.str.lower().str.strip()
                
                # Sample if requested
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    print(f"📋 Sampled {sample_size:,} rows from {len(df):,} total rows")
                
                print(f"✅ Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
                return df
            else:
                raise Exception("Failed to download from kagglehub")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔄 Generating realistic sample data...")
            return self._generate_sample_transaction_data(sample_size or 50000)
    
    def load_credit_data(self, dataset_name="credit_risk", sample_size=None):
        """Load credit data with intelligent fallbacks"""
        try:
            # Try UCI ML repo for credit card default data
            if dataset_name == "credit_default_uci":
                return self._load_uci_credit_data(sample_size)
            
            # Try kagglehub
            file_path = self.download_from_kagglehub(dataset_name)
            
            if file_path and Path(file_path).exists():
                print(f"📊 Loading data from: {file_path}")
                df = pd.read_csv(file_path)
                
                # Standardize column names
                df.columns = df.columns.str.lower().str.strip()
                
                # Sample if requested
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    print(f"📋 Sampled {sample_size:,} rows")
                
                print(f"✅ Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
                return df
            else:
                raise Exception("Failed to download from kagglehub")
                
        except Exception as e:
            print(f"❌ Error loading credit data: {e}")
            print("🔄 Generating realistic sample data...")
            return self._generate_sample_credit_data(sample_size or 10000)
    
    def _load_uci_credit_data(self, sample_size=None):
        """Load credit card default data from UCI ML repository"""
        try:
            print("📥 Loading credit card default data from UCI ML repository...")
            
            # Fetch dataset from UCI
            credit_card_default = fetch_ucirepo(id=350)
            
            # Combine features and targets
            X = credit_card_default.data.features
            y = credit_card_default.data.targets
            
            df = pd.concat([X, y], axis=1)
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Sample if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"📋 Sampled {sample_size:,} rows")
            
            print(f"✅ Loaded UCI credit dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"❌ Error loading UCI data: {e}")
            return self._generate_sample_credit_data(sample_size or 10000)
    
    def _generate_sample_transaction_data(self, n_rows=50000):
        """Generate realistic sample transaction data"""
        print(f"🔄 Generating sample transaction data: {n_rows:,} rows")
        np.random.seed(42)
        
        transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        
        data = {
            'step': np.random.randint(1, 8760, n_rows),
            'type': np.random.choice(transaction_types, n_rows, p=[0.4, 0.2, 0.2, 0.15, 0.05]),
            'amount': np.random.lognormal(mean=4, sigma=2, size=n_rows),
            'nameorig': [f'C{i}' for i in np.random.randint(1, 10000, n_rows)],
            'oldbalanceorg': np.random.exponential(10000, n_rows),
            'newbalanceorig': np.random.exponential(10000, n_rows),
            'namedest': [f'M{i}' if t in ['PAYMENT', 'DEBIT'] else f'C{i}' 
                        for i, t in zip(np.random.randint(1, 10000, n_rows), 
                        np.random.choice(transaction_types, n_rows))],
            'oldbalancedest': np.random.exponential(10000, n_rows),
            'newbalancedest': np.random.exponential(10000, n_rows),
            'isfraud': np.random.choice([0, 1], n_rows, p=[0.9987, 0.0013]),
            'isflaggedfraud': np.random.choice([0, 1], n_rows, p=[0.9999, 0.0001])
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic balance calculations
        df['newbalanceorig'] = np.where(
            df['type'].isin(['CASH_OUT', 'PAYMENT']), 
            np.maximum(0, df['oldbalanceorg'] - df['amount']), 
            df['newbalanceorig']
        )
        
        # Add derived features
        df['hour'] = df['step'] % 24
        df['day'] = df['step'] // 24
        
        return df
    
    def _generate_sample_credit_data(self, n_rows=10000):
        """Generate realistic sample credit data"""
        print(f"🔄 Generating sample credit data: {n_rows:,} rows")
        np.random.seed(42)
        
        data = {
            'id': range(1, n_rows + 1),
            'limit_bal': np.random.uniform(10000, 500000, n_rows),
            'sex': np.random.choice([1, 2], n_rows),  # 1=male, 2=female
            'education': np.random.choice([1, 2, 3, 4], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
            'marriage': np.random.choice([1, 2, 3], n_rows, p=[0.5, 0.4, 0.1]),
            'age': np.random.randint(20, 80, n_rows),
            'default': np.random.choice([0, 1], n_rows, p=[0.78, 0.22])
        }
        
        # Add payment history and bill amounts
        for i in range(6):
            data[f'pay_{i}'] = np.random.choice(
                [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_rows,
                p=[0.6, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]
            )
            data[f'bill_amt{i+1}'] = np.random.uniform(0, 100000, n_rows)
            data[f'pay_amt{i+1}'] = np.random.uniform(0, 50000, n_rows)
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df['total_bill'] = df[[f'bill_amt{i}' for i in range(1, 7)]].sum(axis=1)
        df['total_pay'] = df[[f'pay_amt{i}' for i in range(1, 7)]].sum(axis=1)
        df['utilization_ratio'] = df['total_bill'] / df['limit_bal']
        
        return df
    
    def get_available_datasets(self):
        """Get list of available datasets"""
        return self.datasets
    
    def clear_cache(self):
        """Clear the data cache"""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")

# Global instance for easy import
kaggle_fetcher = KaggleDataFetcher()