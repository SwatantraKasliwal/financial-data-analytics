# Kaggle Data Integration Guide

This guide explains how to set up and use Kaggle datasets directly in your Financial Data Analytics project.

## 📋 Overview

The project now supports automatic data fetching from Kaggle, eliminating the need for manual CSV downloads. The system includes:

- **Automatic dataset detection and download**
- **Intelligent fallback to sample data**
- **Multiple financial datasets support**
- **Caching for faster subsequent runs**

## 🔧 Setup Instructions

### Step 1: Install Dependencies

The required packages are already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Get Kaggle API Credentials

1. **Create a Kaggle Account**: Go to [kaggle.com](https://www.kaggle.com) and sign up

2. **Get API Token**:

   - Go to your Kaggle account settings: [kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll down to the "API" section
   - Click "Create New API Token"
   - Download the `kaggle.json` file

3. **Place Credentials**:
   - **Windows**: Place `kaggle.json` in `C:\Users\<username>\.kaggle\`
   - **Mac/Linux**: Place `kaggle.json` in `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Mac/Linux only)

### Step 3: Alternative Setup Methods

#### Option A: Manual File Placement

```bash
# Create directory
mkdir ~/.kaggle  # Mac/Linux
mkdir C:\Users\%USERNAME%\.kaggle  # Windows

# Copy your kaggle.json file to the directory
```

#### Option B: Environment Variables

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

#### Option C: Using the Streamlit Interface

1. Run your Streamlit app
2. Open the "🔧 Kaggle Configuration" section in the sidebar
3. Enter your username and API key
4. Click "Save Credentials"
5. Restart the app

## 📊 Available Datasets

The system supports these financial datasets from Kaggle:

### 1. PaySim Transactions (Default for Transaction Analysis)

- **Dataset**: `ntnu-testimon/paysim1`
- **File**: `PS_20174392719_1491204439457_log.csv`
- **Size**: ~470MB
- **Description**: Mobile money simulator with fraud detection data
- **Features**: Transaction types, amounts, balances, fraud labels

### 2. Credit Card Fraud Detection

- **Dataset**: `mlg-ulb/creditcardfraud`
- **File**: `creditcard.csv`
- **Size**: ~150MB
- **Description**: Credit card fraud detection with anonymized features

### 3. Credit Risk Assessment (Default for Credit Analysis)

- **Dataset**: `laotse/credit-risk-dataset`
- **File**: `credit_risk_dataset.csv`
- **Size**: ~2MB
- **Description**: Credit risk prediction with borrower characteristics

### 4. Loan Default Prediction

- **Dataset**: `yasserh/loan-default-dataset`
- **File**: `Loan_Default.csv`
- **Size**: ~10MB
- **Description**: Loan default prediction dataset

### 5. Financial Distress Prediction

- **Dataset**: `shebrahimi/financial-distress`
- **File**: `Financial Distress.csv`
- **Size**: ~5MB
- **Description**: Corporate financial distress prediction

## 🚀 Usage Examples

### In Streamlit App

The app automatically fetches data when you run it:

```bash
streamlit run streamlit_app/app.py
```

The data loading process:

1. **Tries Kaggle API first** (if credentials are available)
2. **Falls back to local files** (if they exist)
3. **Generates sample data** (as last resort)

### In Jupyter Notebooks/Python Scripts

```python
from utils.kaggle_data_fetcher import kaggle_fetcher

# Load transaction data (50k sample)
df_transactions = kaggle_fetcher.load_transaction_data(
    dataset_name="paysim_transactions",
    sample_size=50000
)

# Load credit data (10k sample)
df_credit = kaggle_fetcher.load_credit_data(
    dataset_name="credit_risk",
    sample_size=10000
)

# Load full dataset (no sampling)
df_full = kaggle_fetcher.load_transaction_data(
    dataset_name="paysim_transactions",
    sample_size=None
)
```

### Custom Dataset Configuration

```python
from utils.kaggle_data_fetcher import KaggleDataFetcher

# Create custom fetcher
fetcher = KaggleDataFetcher(cache_dir="custom_cache")

# Set up credentials programmatically
fetcher.setup_kaggle_credentials(
    username="your_username",
    key="your_api_key"
)

# Load data
data = fetcher.load_transaction_data("credit_card_fraud")
```

## 🔄 Data Flow

```
1. App/Notebook starts
   ↓
2. Check if Kaggle credentials exist
   ↓
3a. [YES] Try Kaggle API download
   ↓
4a. Cache data locally for future use
   ↓
5. Return processed DataFrame

3b. [NO] Check for local CSV files
   ↓
4b. [Found] Load from local files
   ↓
5b. Return processed DataFrame

4c. [Not Found] Generate sample data
   ↓
5c. Return sample DataFrame
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. "403 Forbidden" Error

```
Solution: Check your Kaggle credentials
- Verify kaggle.json is in the correct location
- Ensure your Kaggle account has API access enabled
- Check that the dataset is publicly available
```

#### 2. "Dataset not found" Error

```
Solution: Verify dataset names
- Check that the dataset exists on Kaggle
- Ensure you have access to the dataset
- Try accessing the dataset through Kaggle web interface first
```

#### 3. "ImportError: No module named 'kaggle'"

```
Solution: Install Kaggle package
pip install kaggle
```

#### 4. Large Dataset Download Issues

```
Solution: Use sampling
- Set sample_size parameter to reduce download size
- Use caching to avoid re-downloading
- Check your internet connection stability
```

### Debug Mode

Enable debug output to see what's happening:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your data loading code
df = kaggle_fetcher.load_transaction_data("paysim_transactions")
```

## 📁 File Structure

After setup, your project structure will look like:

```
financial-data-analytics/
├── utils/
│   ├── __init__.py
│   └── kaggle_data_fetcher.py      # Kaggle integration
├── data_cache/                     # Auto-created cache directory
│   ├── PS_20174392719_1491204439457_log.csv
│   └── credit_risk_dataset.csv
├── streamlit_app/
│   └── app.py                      # Updated with Kaggle support
├── notebooks/
│   └── comprehensive_eda.py        # Updated with Kaggle support
└── requirements.txt                # Includes kaggle package
```

## 🔐 Security Notes

- **Never commit `kaggle.json` to version control**
- **Use environment variables in production**
- **Keep API keys secure and rotate them regularly**
- **The `.gitignore` file should exclude kaggle credentials**

## 🚀 Performance Tips

1. **Use sampling** for development (`sample_size=10000`)
2. **Cache is automatic** - subsequent runs are faster
3. **Clear cache** if you need fresh data: `rm -rf data_cache/`
4. **Use local files** for repeated development work

## 🔄 Fallback Strategy

The system implements a robust fallback strategy:

1. **Kaggle API** (preferred) → Real datasets
2. **Local files** (backup) → Previously downloaded data
3. **Sample generation** (fallback) → Realistic synthetic data

This ensures your application always works, even without internet or Kaggle access.

## 📞 Support

If you encounter issues:

1. Check this documentation first
2. Review the error messages in the console
3. Verify your Kaggle setup at [kaggle.com/settings](https://www.kaggle.com/settings)
4. Test with sample data first before using large datasets

## 🎯 Next Steps

Once setup is complete, you can:

- ✅ Run the Streamlit app with real Kaggle data
- ✅ Execute notebooks with automatic data fetching
- ✅ Experiment with different financial datasets
- ✅ Deploy to cloud platforms with environment variables
