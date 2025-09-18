"""
Kaggle Data Fetcher Utility
Handles downloading and caching data from Kaggle datasets
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import tempfile
import requests
from io import BytesIO
import streamlit as st

class KaggleDataFetcher:
    """Utility class to fetch data from Kaggle datasets"""
    
    def __init__(self, cache_dir="data_cache"):
        """
        Initialize the Kaggle data fetcher
        
        Args:
            cache_dir (str): Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Popular financial datasets from Kaggle
        self.datasets = {
            "paysim_transactions": {
                "dataset": "ntnu-testimon/paysim1",
                "file": "PS_20174392719_1491204439457_log.csv",
                "description": "PaySim Mobile Money Simulator - Financial transaction data",
                "size": "470MB"
            },
            "credit_card_fraud": {
                "dataset": "mlg-ulb/creditcardfraud", 
                "file": "creditcard.csv",
                "description": "Credit Card Fraud Detection Dataset",
                "size": "150MB"
            },
            "financial_distress": {
                "dataset": "shebrahimi/financial-distress",
                "file": "Financial Distress.csv", 
                "description": "Financial Distress Prediction Dataset",
                "size": "5MB"
            },
            "loan_default": {
                "dataset": "yasserh/loan-default-dataset",
                "file": "Loan_Default.csv",
                "description": "Loan Default Prediction Dataset", 
                "size": "10MB"
            },
            "credit_risk": {
                "dataset": "laotse/credit-risk-dataset",
                "file": "credit_risk_dataset.csv",
                "description": "Credit Risk Assessment Dataset",
                "size": "2MB"
            }
        }
    
    def download_dataset_via_api(self, dataset_name):
        """
        Download dataset using Kaggle API (requires kaggle.json credentials)
        
        Args:
            dataset_name (str): Name of the dataset to download
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            import kaggle
            
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found in available datasets")
            
            dataset_info = self.datasets[dataset_name]
            cache_file = self.cache_dir / dataset_info["file"]
            
            # Check if already cached
            if cache_file.exists():
                print(f"✅ Using cached dataset: {cache_file}")
                return str(cache_file)
            
            # Download using Kaggle API
            print(f"📥 Downloading {dataset_name} from Kaggle...")
            kaggle.api.dataset_download_files(
                dataset_info["dataset"], 
                path=str(self.cache_dir),
                unzip=True
            )
            
            if cache_file.exists():
                print(f"✅ Successfully downloaded: {cache_file}")
                return str(cache_file)
            else:
                raise FileNotFoundError(f"Expected file {cache_file} not found after download")
                
        except ImportError:
            raise ImportError("Kaggle API not installed. Install with: pip install kaggle")
        except Exception as e:
            print(f"❌ Error downloading via API: {e}")
            return None
    
    def download_dataset_direct_url(self, dataset_name, backup_url=None):
        """
        Download dataset from direct URL (fallback method)
        
        Args:
            dataset_name (str): Name of the dataset to download
            backup_url (str): Direct URL to download from
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            dataset_info = self.datasets[dataset_name]
            cache_file = self.cache_dir / dataset_info["file"]
            
            # Check if already cached
            if cache_file.exists():
                print(f"✅ Using cached dataset: {cache_file}")
                return str(cache_file)
            
            if not backup_url:
                print("❌ No backup URL provided for direct download")
                return None
            
            print(f"📥 Downloading {dataset_name} from direct URL...")
            response = requests.get(backup_url, stream=True)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Successfully downloaded: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            print(f"❌ Error downloading via direct URL: {e}")
            return None
    
    def load_transaction_data(self, dataset_name="paysim_transactions", sample_size=None):
        """
        Load transaction data from Kaggle
        
        Args:
            dataset_name (str): Name of the dataset to load
            sample_size (int): Number of rows to sample (None for full dataset)
            
        Returns:
            pd.DataFrame: Loaded transaction data
        """
        try:
            # Try API download first
            file_path = self.download_dataset_via_api(dataset_name)
            
            # If API fails, try direct download for some datasets
            if not file_path and dataset_name == "paysim_transactions":
                # Note: You would need actual direct URLs here
                file_path = self.download_dataset_direct_url(
                    dataset_name, 
                    backup_url=None  # Add actual URL if available
                )
            
            if not file_path:
                print("❌ Failed to download dataset, generating sample data...")
                return self._generate_sample_transaction_data(sample_size or 10000)
            
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
            
        except Exception as e:
            print(f"❌ Error loading transaction data: {e}")
            print("🔄 Generating sample data instead...")
            return self._generate_sample_transaction_data(sample_size or 10000)
    
    def load_credit_data(self, dataset_name="credit_risk", sample_size=None):
        """
        Load credit data from Kaggle
        
        Args:
            dataset_name (str): Name of the dataset to load  
            sample_size (int): Number of rows to sample (None for full dataset)
            
        Returns:
            pd.DataFrame: Loaded credit data
        """
        try:
            # Try API download first
            file_path = self.download_dataset_via_api(dataset_name)
            
            if not file_path:
                print("❌ Failed to download dataset, generating sample data...")
                return self._generate_sample_credit_data(sample_size or 5000)
            
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
            
        except Exception as e:
            print(f"❌ Error loading credit data: {e}")
            print("🔄 Generating sample data instead...")
            return self._generate_sample_credit_data(sample_size or 5000)
    
    def _generate_sample_transaction_data(self, n_rows=10000):
        """Generate sample transaction data for fallback"""
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
        df['newbalanceorig'] = np.where(df['type'].isin(['CASH_OUT', 'PAYMENT']), 
                                       df['oldbalanceorg'] - df['amount'], 
                                       df['newbalanceorig'])
        
        print(f"🔄 Generated sample transaction data: {n_rows:,} rows")
        return df
    
    def _generate_sample_credit_data(self, n_rows=5000):
        """Generate sample credit data for fallback"""
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
            data[f'pay_{i}'] = np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_rows, 
                                              p=[0.6, 0.15, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
            data[f'bill_amt{i+1}'] = np.random.uniform(0, 100000, n_rows)
            data[f'pay_amt{i+1}'] = np.random.uniform(0, 50000, n_rows)
        
        df = pd.DataFrame(data)
        print(f"🔄 Generated sample credit data: {n_rows:,} rows")
        return df
    
    def get_available_datasets(self):
        """Get list of available datasets"""
        return self.datasets
    
    def setup_kaggle_credentials(self, username=None, key=None):
        """
        Setup Kaggle API credentials
        
        Args:
            username (str): Kaggle username
            key (str): Kaggle API key
        """
        if username and key:
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_dir.mkdir(exist_ok=True)
            
            credentials = {
                "username": username,
                "key": key
            }
            
            import json
            with open(kaggle_dir / 'kaggle.json', 'w') as f:
                json.dump(credentials, f)
            
            # Set proper permissions on Unix-like systems
            try:
                os.chmod(kaggle_dir / 'kaggle.json', 0o600)
            except:
                pass
            
            print("✅ Kaggle credentials saved successfully!")
        else:
            print("❌ Please provide both username and key")

# Global instance for easy import
kaggle_fetcher = KaggleDataFetcher()