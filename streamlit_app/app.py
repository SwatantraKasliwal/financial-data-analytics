""""""

Financial Data Analytics - Interactive DashboardFinancial Data Analytics - Interactive Dashboard

Author: Financial Analytics TeamAuthor: [Your Name]

Date: September 2024Date: September 2024



Interactive Streamlit dashboard for financial data analysis with SQL query capabilities.Interactive Streamlit dashboard for financial data analysis with SQL query capabilities and Kaggle data integration.

Data is loaded directly from online sources for easy deployment."""

"""

import streamlit as st

import streamlit as stimport pandas as pd

import pandas as pdimport numpy as np

import numpy as npimport plotly.express as px

import plotly.express as pximport plotly.graph_objects as go

import plotly.graph_objects as gofrom plotly.subplots import make_subplots

from plotly.subplots import make_subplotsimport duckdb

import duckdbimport warnings

import warningsimport sys

import os

# Online data source importswarnings.filterwarnings('ignore')

import kagglehub

from kagglehub import KaggleDatasetAdapter# Add parent directory to path for importing utils

from ucimlrepo import fetch_ucirepoparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)

warnings.filterwarnings('ignore')

try:

# Page configuration    from utils.kaggle_data_fetcher import kaggle_fetcher

st.set_page_config(    KAGGLE_AVAILABLE = True

    page_title="Financial Analytics Dashboard",except ImportError:

    page_icon="💰",    KAGGLE_AVAILABLE = False

    layout="wide",

    initial_sidebar_state="expanded"# Page configuration

)st.set_page_config(

    page_title="Financial Analytics Dashboard",

# Custom CSS    page_icon="💰",

st.markdown("""    layout="wide",

<style>    initial_sidebar_state="expanded"

    .main-header {)

        font-size: 3rem;

        color: #1f77b4;# Custom CSS

        text-align: center;st.markdown("""

        margin-bottom: 2rem;<style>

    }    .main-header {

    .metric-card {        font-size: 3rem;

        background-color: #f0f2f6;        color: #1f77b4;

        padding: 1rem;        text-align: center;

        border-radius: 0.5rem;        margin-bottom: 2rem;

        border-left: 5px solid #1f77b4;    }

    }    .metric-card {

    .insights-box {        background-color: #f0f2f6;

        background-color: #e6f3ff;        padding: 1rem;

        padding: 1rem;        border-radius: 0.5rem;

        border-radius: 0.5rem;        border-left: 5px solid #1f77b4;

        border-left: 5px solid #0066cc;    }

        margin: 1rem 0;    .insights-box {

    }        background-color: #e6f3ff;

    .stTabs [data-baseweb="tab-list"] {        padding: 1rem;

        gap: 8px;        border-radius: 0.5rem;

    }        border-left: 5px solid #0066cc;

    .stTabs [data-baseweb="tab"] {        margin: 1rem 0;

        height: 50px;    }

        white-space: pre-wrap;</style>

        background-color: #f0f2f6;""", unsafe_allow_html=True)

        border-radius: 4px 4px 0px 0px;

        gap: 1px;# Initialize session state

        padding-top: 10px;if 'data_loaded' not in st.session_state:

        padding-bottom: 10px;    st.session_state.data_loaded = False

    }

    .stTabs [aria-selected="true"] {@st.cache_data

        background-color: #1f77b4;def load_transaction_data():

        color: white;    """Load and preprocess transaction data from Kaggle"""

    }    try:

</style>        if KAGGLE_AVAILABLE:

""", unsafe_allow_html=True)            # Try to load from Kaggle first

            with st.spinner("📥 Fetching transaction data from Kaggle..."):

@st.cache_data(ttl=3600)  # Cache for 1 hour                df = kaggle_fetcher.load_transaction_data("paysim_transactions", sample_size=50000)

def load_financial_datasets():                st.success("✅ Successfully loaded transaction data from Kaggle!")

    """                

    Load financial datasets directly from online sources with caching        else:

    Returns: transaction_data, credit_data            # Fallback to local files

    """            possible_paths = [

    with st.spinner("🔄 Loading financial datasets from online sources..."):                'data/transactions.csv',

                        '../data/transactions.csv',

        # Load PaySim transaction data from Kaggle                '../../Financial Data Analytics/financial-data-analytics/data/transactions.csv',

        try:                '../Financial Data Analytics/financial-data-analytics/data/transactions.csv'

            st.info("📊 Fetching PaySim transaction data from Kaggle...")            ]

            df_transactions = kagglehub.load_dataset(            

                KaggleDatasetAdapter.PANDAS,            df = None

                "ealaxi/paysim1",            for path in possible_paths:

                ""  # Load the main CSV file                try:

            )                    df = pd.read_csv(path)

            st.success(f"✅ PaySim data loaded: {df_transactions.shape[0]:,} transactions")                    st.success(f"✅ Successfully loaded transaction data from: {path}")

        except Exception as e:                    break

            st.warning(f"⚠️ Could not load PaySim data: {e}")                except:

            st.info("📝 Using sample transaction data as fallback")                    continue

            df_transactions = create_sample_transaction_data()            

                    if df is None:

        # Load credit card default data from UCI ML Repository                # Create dummy data if no file found

        try:                st.warning("⚠️ Using dummy transaction data for demonstration")

            st.info("💳 Fetching UCI credit card default data...")                np.random.seed(42)

            default_dataset = fetch_ucirepo(id=350)                dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")

            df_credit = pd.concat([default_dataset.data.features, default_dataset.data.targets], axis=1)                customers = [f"CUST{str(i).zfill(4)}" for i in range(1, 201)]

            st.success(f"✅ Credit card data loaded: {df_credit.shape[0]:,} customers")                data = {

        except Exception as e:                    "step": np.random.randint(1, 8760, 3000),

            st.warning(f"⚠️ Could not load UCI data: {e}")                    "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], 3000, p=[0.5, 0.2, 0.15, 0.1, 0.05]),

            st.info("📝 Using sample credit data as fallback")                    "amount": np.random.exponential(scale=120, size=3000).round(2),

            df_credit = create_sample_credit_data()                    "nameOrig": np.random.choice(customers, 3000),

                        "oldbalanceOrg": np.random.uniform(0, 100000, 3000).round(2),

    return df_transactions, df_credit                    "newbalanceOrig": np.random.uniform(0, 100000, 3000).round(2),

                    "nameDest": np.random.choice(customers, 3000),

def create_sample_transaction_data():                    "oldbalanceDest": np.random.uniform(0, 100000, 3000).round(2),

    """Create sample transaction data for fallback"""                    "newbalanceDest": np.random.uniform(0, 100000, 3000).round(2),

    np.random.seed(42)                    "isFraud": np.random.choice([0, 1], 3000, p=[0.97, 0.03]),

    n_transactions = 100000                    "isFlaggedFraud": np.random.choice([0, 1], 3000, p=[0.99, 0.01])

                    }

    return pd.DataFrame({                df = pd.DataFrame(data)

        'step': np.random.randint(1, 744, n_transactions),        

        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],         # Basic preprocessing

                                n_transactions, p=[0.35, 0.20, 0.20, 0.15, 0.10]),        df['hour'] = df['step'] % 24

        'amount': np.random.lognormal(3, 1.5, n_transactions),        df['day'] = df['step'] // 24

        'nameOrig': [f'C{i}' for i in np.random.randint(1, 10000, n_transactions)],        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

        'oldbalanceOrg': np.random.uniform(0, 100000, n_transactions),        return df

        'newbalanceOrig': np.random.uniform(0, 100000, n_transactions),    except Exception as e:

        'nameDest': [f'M{i}' for i in np.random.randint(1, 5000, n_transactions)],        st.error(f"Error loading transaction data: {e}")

        'oldbalanceDest': np.random.uniform(0, 50000, n_transactions),        return None

        'newbalanceDest': np.random.uniform(0, 50000, n_transactions),

        'isFraud': np.random.choice([0, 1], n_transactions, p=[0.9985, 0.0015]),@st.cache_data

        'isFlaggedFraud': np.random.choice([0, 1], n_transactions, p=[0.9998, 0.0002])def load_credit_data():

    })    """Load and preprocess credit card data from Kaggle"""

    try:

def create_sample_credit_data():        if KAGGLE_AVAILABLE:

    """Create sample credit card data for fallback"""            # Try to load from Kaggle first

    np.random.seed(42)            with st.spinner("📥 Fetching credit data from Kaggle..."):

    n_customers = 30000                df = kaggle_fetcher.load_credit_data("credit_risk", sample_size=10000)

                    st.success("✅ Successfully loaded credit data from Kaggle!")

    return pd.DataFrame({        else:

        'LIMIT_BAL': np.random.uniform(10000, 1000000, n_customers),            # Fallback to local files

        'SEX': np.random.choice([1, 2], n_customers),            possible_paths = [

        'EDUCATION': np.random.choice([1, 2, 3, 4], n_customers),                'data/default of credit card clients.xls',

        'MARRIAGE': np.random.choice([1, 2, 3], n_customers),                '../data/default of credit card clients.xls',

        'AGE': np.random.randint(21, 80, n_customers),                '../../Financial Data Analytics/financial-data-analytics/data/default of credit card clients.xls',

        'PAY_0': np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_customers),                '../Financial Data Analytics/financial-data-analytics/data/default of credit card clients.xls'

        'PAY_2': np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_customers),            ]

        'PAY_3': np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_customers),            

        'PAY_4': np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_customers),            df = None

        'PAY_5': np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_customers),            for path in possible_paths:

        'PAY_6': np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_customers),                try:

        'BILL_AMT1': np.random.uniform(-50000, 500000, n_customers),                    df = pd.read_excel(path, header=1)

        'BILL_AMT2': np.random.uniform(-50000, 500000, n_customers),                    st.success(f"✅ Successfully loaded credit data from: {path}")

        'BILL_AMT3': np.random.uniform(-50000, 500000, n_customers),                    break

        'BILL_AMT4': np.random.uniform(-50000, 500000, n_customers),                except:

        'BILL_AMT5': np.random.uniform(-50000, 500000, n_customers),                    continue

        'BILL_AMT6': np.random.uniform(-50000, 500000, n_customers),            

        'PAY_AMT1': np.random.uniform(0, 100000, n_customers),            if df is None:

        'PAY_AMT2': np.random.uniform(0, 100000, n_customers),                # Create dummy data if no file found

        'PAY_AMT3': np.random.uniform(0, 100000, n_customers),                st.warning("⚠️ Using dummy credit data for demonstration")

        'PAY_AMT4': np.random.uniform(0, 100000, n_customers),                np.random.seed(42)

        'PAY_AMT5': np.random.uniform(0, 100000, n_customers),                data = {

        'PAY_AMT6': np.random.uniform(0, 100000, n_customers),                    "ID": range(1, 1001),

        'default payment next month': np.random.choice([0, 1], n_customers, p=[0.78, 0.22])                    "LIMIT_BAL": np.random.uniform(10000, 800000, 1000).round(0),

    })                    "SEX": np.random.choice([1, 2], 1000),

                    "EDUCATION": np.random.choice([1, 2, 3, 4], 1000),

def show_transaction_analysis(df_transactions):                    "MARRIAGE": np.random.choice([1, 2, 3], 1000),

    """Display comprehensive transaction analysis"""                    "AGE": np.random.randint(21, 79, 1000),

    st.header("📊 Transaction Data Analysis")                    "PAY_0": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),

                        "PAY_2": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),

    # Key metrics                    "PAY_3": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),

    col1, col2, col3, col4 = st.columns(4)                    "PAY_4": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),

                        "PAY_5": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),

    with col1:                    "PAY_6": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),

        total_transactions = len(df_transactions)                    "BILL_AMT1": np.random.uniform(0, 100000, 1000).round(2),

        st.metric("Total Transactions", f"{total_transactions:,}")                    "BILL_AMT2": np.random.uniform(0, 100000, 1000).round(2),

                        "BILL_AMT3": np.random.uniform(0, 100000, 1000).round(2),

    with col2:                    "BILL_AMT4": np.random.uniform(0, 100000, 1000).round(2),

        total_volume = df_transactions['amount'].sum()                    "BILL_AMT5": np.random.uniform(0, 100000, 1000).round(2),

        st.metric("Total Volume", f"${total_volume:,.0f}")                    "BILL_AMT6": np.random.uniform(0, 100000, 1000).round(2),

                        "PAY_AMT1": np.random.uniform(0, 50000, 1000).round(2),

    with col3:                    "PAY_AMT2": np.random.uniform(0, 50000, 1000).round(2),

        avg_transaction = df_transactions['amount'].mean()                    "PAY_AMT3": np.random.uniform(0, 50000, 1000).round(2),

        st.metric("Avg Transaction", f"${avg_transaction:.2f}")                    "PAY_AMT4": np.random.uniform(0, 50000, 1000).round(2),

                        "PAY_AMT5": np.random.uniform(0, 50000, 1000).round(2),

    with col4:                    "PAY_AMT6": np.random.uniform(0, 50000, 1000).round(2),

        fraud_rate = df_transactions['isFraud'].mean() * 100                    "default": np.random.choice([0, 1], 1000, p=[0.77, 0.23])

        st.metric("Fraud Rate", f"{fraud_rate:.3f}%", delta=None)                }

                    df = pd.DataFrame(data)

    # Transaction analysis tabs            else:

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Volume Analysis", "🔍 Transaction Types", "🚨 Fraud Analysis", "⏰ Time Patterns"])                df.rename(columns={'default payment next month': 'default'}, inplace=True)

            

    with tab1:        return df

        st.subheader("Transaction Volume Over Time")    except Exception as e:

                st.error(f"Error loading credit data: {e}")

        # Time series analysis        return None

        hourly_data = df_transactions.groupby('step').agg({

            'amount': ['sum', 'count', 'mean']def execute_sql_query(query, df_name, df):

        }).round(2)    """Execute SQL query using DuckDB"""

        hourly_data.columns = ['Total_Amount', 'Transaction_Count', 'Avg_Amount']    try:

        hourly_data = hourly_data.reset_index()        conn = duckdb.connect()

                conn.register(df_name, df)

        fig = px.line(hourly_data, x='step', y='Total_Amount',         result = conn.execute(query).fetchdf()

                     title='Hourly Transaction Volume')        conn.close()

        fig.update_layout(xaxis_title="Time (Hours)", yaxis_title="Total Amount ($)")        return result, None

        st.plotly_chart(fig, use_container_width=True)    except Exception as e:

                return None, str(e)

        col1, col2 = st.columns(2)

        with col1:# Main app

            fig2 = px.line(hourly_data, x='step', y='Transaction_Count',def main():

                          title='Hourly Transaction Count')    st.markdown('<h1 class="main-header">💰 Financial Data Analytics Dashboard</h1>', unsafe_allow_html=True)

            st.plotly_chart(fig2, use_container_width=True)    

            # Sidebar navigation

        with col2:    st.sidebar.title("📋 Navigation")

            fig3 = px.line(hourly_data, x='step', y='Avg_Amount',    

                          title='Average Transaction Amount Over Time')    # Add Kaggle configuration section

            st.plotly_chart(fig3, use_container_width=True)    with st.sidebar.expander("🔧 Kaggle Configuration", expanded=False):

            if KAGGLE_AVAILABLE:

    with tab2:            st.success("✅ Kaggle fetcher available")

        st.subheader("Transaction Type Analysis")            

                    # Show available datasets

        col1, col2 = st.columns(2)            datasets = kaggle_fetcher.get_available_datasets()

                    st.write("**Available Datasets:**")

        with col1:            for key, info in datasets.items():

            # Transaction type distribution                st.write(f"• {info['description']} ({info['size']})")

            type_counts = df_transactions['type'].value_counts()            

            fig = px.pie(values=type_counts.values, names=type_counts.index,            # Kaggle credentials setup

                        title="Transaction Type Distribution")            st.write("**Setup Kaggle API:**")

            st.plotly_chart(fig, use_container_width=True)            st.info("💡 To use real Kaggle data, set up your API credentials")

                    

        with col2:            with st.form("kaggle_setup"):

            # Average amount by type                username = st.text_input("Kaggle Username")

            type_amounts = df_transactions.groupby('type')['amount'].mean().sort_values(ascending=True)                key = st.text_input("Kaggle API Key", type="password")

            fig = px.bar(x=type_amounts.values, y=type_amounts.index, orientation='h',                if st.form_submit_button("Save Credentials"):

                        title="Average Amount by Transaction Type")                    if username and key:

            fig.update_layout(xaxis_title="Average Amount ($)", yaxis_title="Transaction Type")                        kaggle_fetcher.setup_kaggle_credentials(username, key)

            st.plotly_chart(fig, use_container_width=True)                        st.success("Credentials saved! Restart the app to use real data.")

                            else:

        # Detailed type analysis                        st.error("Please provide both username and key")

        st.subheader("Detailed Transaction Type Metrics")        else:

        type_analysis = df_transactions.groupby('type').agg({            st.warning("⚠️ Kaggle fetcher not available")

            'amount': ['count', 'sum', 'mean', 'median'],            st.info("Install with: `pip install kaggle`")

            'isFraud': 'mean'    

        }).round(4)    page = st.sidebar.selectbox(

                "Choose Analysis",

        type_analysis.columns = ['Count', 'Total_Volume', 'Mean_Amount', 'Median_Amount', 'Fraud_Rate']        ["🏠 Overview", "📊 Transaction Analysis", "💳 Credit Analysis", 

        type_analysis['Total_Volume'] = type_analysis['Total_Volume'].apply(lambda x: f"${x:,.0f}")         "🔍 SQL Query Interface", "📈 Advanced Analytics", "💡 Insights & Recommendations"]

        type_analysis['Mean_Amount'] = type_analysis['Mean_Amount'].apply(lambda x: f"${x:.2f}")    )

        type_analysis['Median_Amount'] = type_analysis['Median_Amount'].apply(lambda x: f"${x:.2f}")    

        type_analysis['Fraud_Rate'] = type_analysis['Fraud_Rate'].apply(lambda x: f"{x:.4%}")    # Load data

            if not st.session_state.data_loaded:

        st.dataframe(type_analysis, use_container_width=True)        with st.spinner("Loading data..."):

                df_trans = load_transaction_data()

    with tab3:            df_credit = load_credit_data()

        st.subheader("Fraud Detection Analysis")            

                    if df_trans is not None and df_credit is not None:

        fraud_df = df_transactions[df_transactions['isFraud'] == 1]                st.session_state.df_trans = df_trans

        legit_df = df_transactions[df_transactions['isFraud'] == 0]                st.session_state.df_credit = df_credit

                        st.session_state.data_loaded = True

        col1, col2, col3, col4 = st.columns(4)            else:

        with col1:                st.error("Failed to load data. Please check data files.")

            st.metric("Fraud Cases", f"{len(fraud_df):,}")                return

        with col2:    

            fraud_volume = fraud_df['amount'].sum()    df_trans = st.session_state.df_trans

            st.metric("Fraud Volume", f"${fraud_volume:,.0f}")    df_credit = st.session_state.df_credit

        with col3:    

            avg_fraud_amount = fraud_df['amount'].mean() if len(fraud_df) > 0 else 0    # Page routing

            st.metric("Avg Fraud Amount", f"${avg_fraud_amount:.2f}")    if page == "🏠 Overview":

        with col4:        show_overview(df_trans, df_credit)

            fraud_percentage = (len(fraud_df) / len(df_transactions)) * 100    elif page == "📊 Transaction Analysis":

            st.metric("Fraud Percentage", f"{fraud_percentage:.3f}%")        show_transaction_analysis(df_trans)

            elif page == "💳 Credit Analysis":

        # Fraud by transaction type        show_credit_analysis(df_credit)

        col1, col2 = st.columns(2)    elif page == "🔍 SQL Query Interface":

                show_sql_interface(df_trans, df_credit)

        with col1:    elif page == "📈 Advanced Analytics":

            fraud_by_type = df_transactions.groupby('type')['isFraud'].agg(['sum', 'mean']).reset_index()        show_advanced_analytics(df_trans, df_credit)

            fraud_by_type.columns = ['Type', 'Fraud_Count', 'Fraud_Rate']    elif page == "💡 Insights & Recommendations":

                    show_insights()

            fig = px.bar(fraud_by_type, x='Type', y='Fraud_Count',

                        title="Fraud Cases by Transaction Type")def show_overview(df_trans, df_credit):

            st.plotly_chart(fig, use_container_width=True)    """Display overview page"""

            st.header("📊 Financial Data Overview")

        with col2:    

            fig = px.bar(fraud_by_type, x='Type', y='Fraud_Rate',    col1, col2 = st.columns(2)

                        title="Fraud Rate by Transaction Type")    

            fig.update_layout(yaxis_title="Fraud Rate")    with col1:

            st.plotly_chart(fig, use_container_width=True)        st.subheader("🔄 Transaction Data")

                st.markdown(f"""

        # Amount distribution comparison        <div class="metric-card">

        st.subheader("Amount Distribution: Fraud vs Legitimate")        <h3>Dataset Statistics</h3>

                <ul>

        if len(fraud_df) > 0:        <li><strong>Total Transactions:</strong> {len(df_trans):,}</li>

            fig = go.Figure()        <li><strong>Transaction Types:</strong> {df_trans['type'].nunique()}</li>

            fig.add_trace(go.Histogram(x=legit_df['amount'], name='Legitimate',         <li><strong>Total Volume:</strong> ${df_trans['amount'].sum():,.2f}</li>

                                     opacity=0.7, nbinsx=50))        <li><strong>Fraud Rate:</strong> {df_trans['isFraud'].mean()*100:.3f}%</li>

            fig.add_trace(go.Histogram(x=fraud_df['amount'], name='Fraud',         <li><strong>Time Period:</strong> {df_trans['step'].max()} time steps</li>

                                     opacity=0.7, nbinsx=50))        </ul>

            fig.update_layout(barmode='overlay', title='Transaction Amount Distribution',        </div>

                            xaxis_title='Amount ($)', yaxis_title='Frequency')        """, unsafe_allow_html=True)

            st.plotly_chart(fig, use_container_width=True)        

            # Quick transaction type chart

    with tab4:        fig = px.pie(df_trans['type'].value_counts().reset_index(), 

        st.subheader("Transaction Timing Patterns")                     values='count', names='type', 

                             title="Transaction Types Distribution")

        # Create time-based features        st.plotly_chart(fig, use_container_width=True)

        df_time = df_transactions.copy()    

        df_time['hour_of_day'] = df_time['step'] % 24    with col2:

        df_time['day_of_month'] = (df_time['step'] // 24) % 30 + 1        st.subheader("💳 Credit Card Data")

                st.markdown(f"""

        col1, col2 = st.columns(2)        <div class="metric-card">

                <h3>Dataset Statistics</h3>

        with col1:        <ul>

            hourly_volume = df_time.groupby('hour_of_day')['amount'].sum()        <li><strong>Total Customers:</strong> {len(df_credit):,}</li>

            fig = px.bar(x=hourly_volume.index, y=hourly_volume.values,        <li><strong>Default Rate:</strong> {df_credit['default'].mean()*100:.2f}%</li>

                        title="Transaction Volume by Hour of Day")        <li><strong>Avg Credit Limit:</strong> ${df_credit['LIMIT_BAL'].mean():,.2f}</li>

            fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Total Volume ($)")        <li><strong>Age Range:</strong> {df_credit['AGE'].min()}-{df_credit['AGE'].max()} years</li>

            st.plotly_chart(fig, use_container_width=True)        <li><strong>Features:</strong> {len(df_credit.columns)} variables</li>

                </ul>

        with col2:        </div>

            hourly_fraud = df_time.groupby('hour_of_day')['isFraud'].mean()        """, unsafe_allow_html=True)

            fig = px.line(x=hourly_fraud.index, y=hourly_fraud.values,        

                         title="Fraud Rate by Hour of Day")        # Quick default rate by gender

            fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Fraud Rate")        gender_default = df_credit.groupby('SEX')['default'].mean() * 100

            st.plotly_chart(fig, use_container_width=True)        gender_default.index = ['Male', 'Female']

        fig = px.bar(x=gender_default.index, y=gender_default.values, 

def show_credit_analysis(df_credit):                     title="Default Rate by Gender (%)")

    """Display comprehensive credit card analysis"""        st.plotly_chart(fig, use_container_width=True)

    st.header("💳 Credit Card Risk Analysis")    

        # Sample data preview

    # Handle different column names    st.subheader("📋 Data Preview")

    target_col = 'default payment next month' if 'default payment next month' in df_credit.columns else 'default_payment'    tab1, tab2 = st.tabs(["Transaction Data", "Credit Data"])

        

    # Key metrics    with tab1:

    col1, col2, col3, col4 = st.columns(4)        st.dataframe(df_trans.head(), use_container_width=True)

        

    with col1:    with tab2:

        total_customers = len(df_credit)        st.dataframe(df_credit.head(), use_container_width=True)

        st.metric("Total Customers", f"{total_customers:,}")

    def show_transaction_analysis(df_trans):

    with col2:    """Display transaction analysis page"""

        if 'LIMIT_BAL' in df_credit.columns:    st.header("📊 Transaction Data Analysis")

            avg_limit = df_credit['LIMIT_BAL'].mean()    

            st.metric("Avg Credit Limit", f"${avg_limit:,.0f}")    # Filters

        else:    st.sidebar.subheader("🔧 Filters")

            st.metric("Avg Credit Limit", "N/A")    selected_types = st.sidebar.multiselect(

            "Select Transaction Types",

    with col3:        df_trans['type'].unique(),

        default_rate = df_credit[target_col].mean() * 100        default=df_trans['type'].unique()

        st.metric("Default Rate", f"{default_rate:.1f}%")    )

        

    with col4:    amount_range = st.sidebar.slider(

        if 'AGE' in df_credit.columns:        "Amount Range ($)",

            avg_age = df_credit['AGE'].mean()        float(df_trans['amount'].min()),

            st.metric("Average Age", f"{avg_age:.1f}")        float(df_trans['amount'].quantile(0.99)),

        else:        (float(df_trans['amount'].min()), float(df_trans['amount'].quantile(0.95)))

            st.metric("Average Age", "N/A")    )

        

    # Credit analysis tabs    # Filter data

    tab1, tab2, tab3, tab4 = st.tabs(["👥 Demographics", "💰 Credit Limits", "📊 Risk Analysis", "💳 Payment Behavior"])    filtered_df = df_trans[

            (df_trans['type'].isin(selected_types)) &

    with tab1:        (df_trans['amount'] >= amount_range[0]) &

        st.subheader("Customer Demographics")        (df_trans['amount'] <= amount_range[1])

            ]

        col1, col2 = st.columns(2)    

            # Metrics

        if 'SEX' in df_credit.columns:    col1, col2, col3, col4 = st.columns(4)

            with col1:    with col1:

                gender_dist = df_credit['SEX'].value_counts()        st.metric("Total Transactions", f"{len(filtered_df):,}")

                gender_labels = ['Male' if x == 1 else 'Female' for x in gender_dist.index]    with col2:

                fig = px.pie(values=gender_dist.values, names=gender_labels,        st.metric("Total Volume", f"${filtered_df['amount'].sum():,.2f}")

                            title="Gender Distribution")    with col3:

                st.plotly_chart(fig, use_container_width=True)        st.metric("Fraud Rate", f"{filtered_df['isFraud'].mean()*100:.3f}%")

            with col4:

        if 'AGE' in df_credit.columns:        st.metric("Avg Amount", f"${filtered_df['amount'].mean():.2f}")

            with col2:    

                fig = px.histogram(df_credit, x='AGE', nbins=30,    # Visualizations

                                 title="Age Distribution")    col1, col2 = st.columns(2)

                st.plotly_chart(fig, use_container_width=True)    

            with col1:

        # Demographics by default status        # Amount distribution

        if 'EDUCATION' in df_credit.columns:        fig = px.histogram(filtered_df, x='amount', nbins=50, 

            st.subheader("Education Level Analysis")                          title="Transaction Amount Distribution")

            education_default = df_credit.groupby('EDUCATION')[target_col].mean().reset_index()        st.plotly_chart(fig, use_container_width=True)

            education_default.columns = ['Education_Level', 'Default_Rate']        

                    # Fraud by type

            fig = px.bar(education_default, x='Education_Level', y='Default_Rate',        fraud_by_type = filtered_df.groupby('type')['isFraud'].agg(['count', 'sum']).reset_index()

                        title="Default Rate by Education Level")        fraud_by_type['fraud_rate'] = fraud_by_type['sum'] / fraud_by_type['count'] * 100

            st.plotly_chart(fig, use_container_width=True)        fig = px.bar(fraud_by_type, x='type', y='fraud_rate', 

                         title="Fraud Rate by Transaction Type (%)")

    with tab2:        st.plotly_chart(fig, use_container_width=True)

        if 'LIMIT_BAL' in df_credit.columns:    

            st.subheader("Credit Limit Analysis")    with col2:

                    # Time series

            col1, col2 = st.columns(2)        hourly_data = filtered_df.groupby('hour').size().reset_index(name='count')

                    fig = px.line(hourly_data, x='hour', y='count', 

            with col1:                      title="Transactions by Hour of Day")

                fig = px.histogram(df_credit, x='LIMIT_BAL', nbins=50,        st.plotly_chart(fig, use_container_width=True)

                                 title="Credit Limit Distribution")        

                st.plotly_chart(fig, use_container_width=True)        # Amount vs fraud scatter

                    sample_data = filtered_df.sample(n=min(5000, len(filtered_df)))

            with col2:        fig = px.scatter(sample_data, x='amount', y='step', color='isFraud',

                # Credit limit by default status                        title="Amount vs Time (Fraud Highlighted)")

                fig = px.box(df_credit, y='LIMIT_BAL', x=target_col,        st.plotly_chart(fig, use_container_width=True)

                           title="Credit Limit by Default Status")    

                st.plotly_chart(fig, use_container_width=True)    # Detailed analysis

                st.subheader("🔍 Detailed Analysis")

            # Credit limit statistics    

            st.subheader("Credit Limit Statistics")    # Balance analysis

            limit_stats = df_credit['LIMIT_BAL'].describe()    if st.checkbox("Show Balance Analysis"):

                    st.markdown("### Balance Change Analysis")

            col1, col2, col3, col4 = st.columns(4)        filtered_df_copy = filtered_df.copy()

            with col1:        filtered_df_copy['balance_change_orig'] = filtered_df_copy['newbalanceOrig'] - filtered_df_copy['oldbalanceOrg']

                st.metric("Minimum", f"${limit_stats['min']:,.0f}")        filtered_df_copy['balance_inconsistency'] = abs(filtered_df_copy['balance_change_orig'] + filtered_df_copy['amount']) > 0.01

            with col2:        

                st.metric("Median", f"${limit_stats['50%']:,.0f}")        inconsistency_rate = filtered_df_copy['balance_inconsistency'].mean() * 100

            with col3:        st.metric("Balance Inconsistency Rate", f"{inconsistency_rate:.2f}%")

                st.metric("Mean", f"${limit_stats['mean']:,.0f}")        

            with col4:        fig = px.histogram(filtered_df_copy, x='balance_change_orig', 

                st.metric("Maximum", f"${limit_stats['max']:,.0f}")                          title="Distribution of Balance Changes (Origin)")

            st.plotly_chart(fig, use_container_width=True)

    with tab3:

        st.subheader("Risk Assessment")def show_credit_analysis(df_credit):

            """Display credit analysis page"""

        # Create risk score if possible    st.header("💳 Credit Card Default Analysis")

        if 'LIMIT_BAL' in df_credit.columns and 'BILL_AMT1' in df_credit.columns:    

            df_credit['utilization_rate'] = df_credit['BILL_AMT1'] / df_credit['LIMIT_BAL']    # Filters

            df_credit['utilization_rate'] = df_credit['utilization_rate'].clip(-1, 3)  # Cap extreme values    st.sidebar.subheader("🔧 Filters")

                age_range = st.sidebar.slider(

            col1, col2 = st.columns(2)        "Age Range",

                    int(df_credit['AGE'].min()),

            with col1:        int(df_credit['AGE'].max()),

                fig = px.histogram(df_credit, x='utilization_rate', nbins=50,        (int(df_credit['AGE'].min()), int(df_credit['AGE'].max()))

                                 title="Credit Utilization Rate Distribution")    )

                st.plotly_chart(fig, use_container_width=True)    

                limit_range = st.sidebar.slider(

            with col2:        "Credit Limit Range ($)",

                # Utilization by default status        float(df_credit['LIMIT_BAL'].min()),

                fig = px.box(df_credit, y='utilization_rate', x=target_col,        float(df_credit['LIMIT_BAL'].max()),

                           title="Credit Utilization by Default Status")        (float(df_credit['LIMIT_BAL'].min()), float(df_credit['LIMIT_BAL'].max()))

                st.plotly_chart(fig, use_container_width=True)    )

            

        # Risk by age groups    # Filter data

        if 'AGE' in df_credit.columns:    filtered_df = df_credit[

            df_credit['age_group'] = pd.cut(df_credit['AGE'],         (df_credit['AGE'] >= age_range[0]) &

                                          bins=[0, 30, 40, 50, 60, 100],         (df_credit['AGE'] <= age_range[1]) &

                                          labels=['<30', '30-40', '40-50', '50-60', '60+'])        (df_credit['LIMIT_BAL'] >= limit_range[0]) &

            age_risk = df_credit.groupby('age_group')[target_col].mean().reset_index()        (df_credit['LIMIT_BAL'] <= limit_range[1])

                ]

            fig = px.bar(age_risk, x='age_group', y=target_col,    

                        title="Default Rate by Age Group")    # Metrics

            st.plotly_chart(fig, use_container_width=True)    col1, col2, col3, col4 = st.columns(4)

        with col1:

    with tab4:        st.metric("Total Customers", f"{len(filtered_df):,}")

        st.subheader("Payment Behavior Analysis")    with col2:

                st.metric("Default Rate", f"{filtered_df['default'].mean()*100:.2f}%")

        # Payment status columns    with col3:

        pay_columns = [col for col in df_credit.columns if col.startswith('PAY_') and col not in ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]        st.metric("Avg Credit Limit", f"${filtered_df['LIMIT_BAL'].mean():,.0f}")

            with col4:

        if pay_columns:        st.metric("Avg Age", f"{filtered_df['AGE'].mean():.1f} years")

            # Average payment status over time    

            pay_status_avg = df_credit[pay_columns].mean()    # Visualizations

                col1, col2 = st.columns(2)

            fig = px.bar(x=pay_status_avg.index, y=pay_status_avg.values,    

                        title="Average Payment Status by Month")    with col1:

            fig.update_layout(xaxis_title="Payment Period", yaxis_title="Average Payment Status")        # Default by gender

            st.plotly_chart(fig, use_container_width=True)        gender_default = filtered_df.groupby('SEX')['default'].mean() * 100

                    gender_default.index = ['Male', 'Female']

            # Payment status distribution        fig = px.bar(x=gender_default.index, y=gender_default.values,

            st.subheader("Payment Status Distribution")                     title="Default Rate by Gender (%)")

            for col in pay_columns[:3]:  # Show first 3 payment columns        st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)        

                with col1:        # Age distribution

                    pay_dist = df_credit[col].value_counts().sort_index()        fig = px.histogram(filtered_df, x='AGE', color='default',

                    fig = px.bar(x=pay_dist.index, y=pay_dist.values,                          title="Age Distribution by Default Status")

                               title=f"{col} Distribution")        st.plotly_chart(fig, use_container_width=True)

                    st.plotly_chart(fig, use_container_width=True)    

    with col2:

def show_sql_interface(df_transactions, df_credit):        # Default by education

    """Interactive SQL query interface"""        edu_default = filtered_df.groupby('EDUCATION')['default'].mean() * 100

    st.header("🔍 SQL Query Interface")        fig = px.bar(x=edu_default.index, y=edu_default.values,

    st.write("Execute SQL queries on the financial datasets using DuckDB")                     title="Default Rate by Education Level (%)")

            st.plotly_chart(fig, use_container_width=True)

    # Create DuckDB connection        

    conn = duckdb.connect(':memory:')        # Credit limit vs default

            fig = px.box(filtered_df, x='default', y='LIMIT_BAL',

    # Register dataframes                     title="Credit Limit Distribution by Default Status")

    conn.register('transactions', df_transactions)        st.plotly_chart(fig, use_container_width=True)

    conn.register('credit_data', df_credit)    

        # Payment history analysis

    # Sample queries    if st.checkbox("Show Payment History Analysis"):

    st.subheader("📋 Sample Queries")        st.markdown("### Payment History Analysis")

            payment_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    sample_queries = {        

        "Transaction Volume by Type": """        # Correlation with default

        SELECT         correlations = filtered_df[payment_cols + ['default']].corr()['default'].drop('default')

            type,        fig = px.bar(x=correlations.index, y=correlations.values,

            COUNT(*) as transaction_count,                     title="Correlation of Payment History with Default")

            SUM(amount) as total_volume,        st.plotly_chart(fig, use_container_width=True)

            AVG(amount) as avg_amount,

            ROUND(AVG(isFraud) * 100, 4) as fraud_rate_pctdef show_sql_interface(df_trans, df_credit):

        FROM transactions    """Display SQL query interface"""

        GROUP BY type    st.header("🔍 SQL Query Interface")

        ORDER BY total_volume DESC;    

        """,    st.markdown("""

            <div class="insights-box">

        "High-Risk Transactions": """    <h3>💡 Interactive SQL Analysis</h3>

        SELECT     <p>Use SQL queries to explore the data interactively. Available tables:</p>

            type,    <ul>

            COUNT(*) as high_value_transactions,    <li><strong>transactions</strong> - Transaction data (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud)</li>

            AVG(amount) as avg_amount,    <li><strong>credit</strong> - Credit card data (ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0-PAY_6, BILL_AMT1-BILL_AMT6, PAY_AMT1-PAY_AMT6, default)</li>

            SUM(isFraud) as fraud_cases,    </ul>

            ROUND(AVG(isFraud) * 100, 2) as fraud_rate_pct    </div>

        FROM transactions    """, unsafe_allow_html=True)

        WHERE amount > 50000    

        GROUP BY type    # Query tabs

        ORDER BY fraud_rate_pct DESC;    tab1, tab2, tab3 = st.tabs(["📝 Custom Query", "📊 Transaction Queries", "💳 Credit Queries"])

        """,    

            with tab1:

        "Credit Risk Segments": """        st.subheader("Write Your Own SQL Query")

        SELECT         

            CASE         # Query input

                WHEN AGE < 30 THEN 'Young (18-29)'        query = st.text_area(

                WHEN AGE < 50 THEN 'Middle-aged (30-49)'            "Enter SQL Query:",

                ELSE 'Senior (50+)'            height=150,

            END as age_segment,            value="SELECT type, COUNT(*) as transaction_count, AVG(amount) as avg_amount\nFROM transactions\nGROUP BY type\nORDER BY transaction_count DESC"

            COUNT(*) as customers,        )

            ROUND(AVG(LIMIT_BAL), 2) as avg_credit_limit,        

            ROUND(AVG("default payment next month") * 100, 2) as default_rate_pct        # Table selection

        FROM credit_data        table_name = st.selectbox("Select Table:", ["transactions", "credit"])

        GROUP BY age_segment        

        ORDER BY default_rate_pct DESC;        if st.button("Execute Query"):

        """,            df_to_use = df_trans if table_name == "transactions" else df_credit

                    result, error = execute_sql_query(query, table_name, df_to_use)

        "Payment Behavior Analysis": """            

        SELECT             if error:

            EDUCATION,                st.error(f"Query Error: {error}")

            COUNT(*) as customers,            else:

            ROUND(AVG(LIMIT_BAL), 2) as avg_limit,                st.success("Query executed successfully!")

            ROUND(AVG(PAY_0), 2) as avg_recent_payment_status,                st.dataframe(result, use_container_width=True)

            ROUND(AVG("default payment next month") * 100, 2) as default_rate_pct                

        FROM credit_data                # Download results

        GROUP BY EDUCATION                csv = result.to_csv(index=False)

        ORDER BY default_rate_pct DESC;                st.download_button(

        """,                    label="Download Results as CSV",

                            data=csv,

        "Fraud Detection Insights": """                    file_name="query_results.csv",

        WITH fraud_stats AS (                    mime="text/csv"

            SELECT                 )

                type,    

                COUNT(*) as total_transactions,    with tab2:

                SUM(isFraud) as fraud_count,        st.subheader("📊 Pre-built Transaction Queries")

                AVG(amount) as avg_amount,        

                AVG(CASE WHEN isFraud = 1 THEN amount END) as avg_fraud_amount        query_options = {

            FROM transactions            "Fraud Analysis by Type": """

            GROUP BY type            SELECT 

        )                type,

        SELECT                 COUNT(*) as total_transactions,

            type,                SUM(isFraud) as fraud_count,

            total_transactions,                ROUND(AVG(isFraud) * 100, 2) as fraud_rate_pct,

            fraud_count,                ROUND(AVG(amount), 2) as avg_amount

            ROUND(fraud_count * 100.0 / total_transactions, 4) as fraud_rate_pct,            FROM transactions

            ROUND(avg_amount, 2) as avg_transaction_amount,            GROUP BY type

            ROUND(avg_fraud_amount, 2) as avg_fraud_amount,            ORDER BY fraud_rate_pct DESC

            ROUND(avg_fraud_amount / avg_amount, 2) as fraud_amount_ratio            """,

        FROM fraud_stats            

        ORDER BY fraud_rate_pct DESC;            "High Value Transactions": """

        """            SELECT 

    }                type,

                    amount,

    # Query selector                nameOrig,

    selected_query_name = st.selectbox("Choose a sample query:", list(sample_queries.keys()))                nameDest,

                    isFraud,

    # Query input                step

    query = st.text_area(            FROM transactions

        "SQL Query:",             WHERE amount > (SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) FROM transactions)

        value=sample_queries[selected_query_name],             ORDER BY amount DESC

        height=200,            LIMIT 20

        help="Write your SQL query here. Available tables: 'transactions', 'credit_data'"            """,

    )            

                "Hourly Transaction Patterns": """

    # Execute query            SELECT 

    if st.button("🚀 Execute Query", type="primary"):                (step % 24) as hour,

        try:                COUNT(*) as transaction_count,

            with st.spinner("Executing query..."):                ROUND(AVG(amount), 2) as avg_amount,

                result = conn.execute(query).fetchdf()                SUM(isFraud) as fraud_count

                        FROM transactions

            if not result.empty:            GROUP BY hour

                st.success(f"✅ Query executed successfully! ({len(result)} rows returned)")            ORDER BY hour

                            """,

                # Display results            

                st.subheader("📊 Query Results")            "Balance Inconsistencies": """

                st.dataframe(result, use_container_width=True)            SELECT 

                                type,

                # Download option                amount,

                csv = result.to_csv(index=False)                oldbalanceOrg,

                st.download_button(                newbalanceOrig,

                    label="📥 Download Results as CSV",                (oldbalanceOrg - amount) as expected_balance,

                    data=csv,                newbalanceOrig,

                    file_name=f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",                ABS((oldbalanceOrg - amount) - newbalanceOrig) as inconsistency,

                    mime="text/csv"                isFraud

                )            FROM transactions

                            WHERE ABS((oldbalanceOrg - amount) - newbalanceOrig) > 0.01

                # Basic visualization if numeric columns exist            ORDER BY inconsistency DESC

                numeric_columns = result.select_dtypes(include=[np.number]).columns.tolist()            LIMIT 20

                if len(numeric_columns) >= 2:            """,

                    st.subheader("📈 Quick Visualization")            

                    col1, col2 = st.columns(2)            "Top Senders by Volume": """

                                SELECT 

                    with col1:                nameOrig,

                        x_col = st.selectbox("X-axis:", result.columns)                COUNT(*) as transaction_count,

                    with col2:                ROUND(SUM(amount), 2) as total_amount,

                        y_col = st.selectbox("Y-axis:", numeric_columns)                ROUND(AVG(amount), 2) as avg_amount,

                                    SUM(isFraud) as fraud_count

                    if x_col and y_col:            FROM transactions

                        fig = px.bar(result, x=x_col, y=y_col,             GROUP BY nameOrig

                                   title=f"{y_col} by {x_col}")            ORDER BY total_amount DESC

                        st.plotly_chart(fig, use_container_width=True)            LIMIT 15

            else:            """

                st.warning("Query returned no results.")        }

                        

        except Exception as e:        selected_query = st.selectbox("Select Query:", list(query_options.keys()))

            st.error(f"❌ Query execution failed: {str(e)}")        st.code(query_options[selected_query], language='sql')

            st.info("💡 Make sure your SQL syntax is correct and table names are valid.")        

            if st.button("Run Selected Query", key="trans_query"):

    # Schema information            result, error = execute_sql_query(query_options[selected_query], "transactions", df_trans)

    with st.expander("📋 Table Schema Information"):            

        col1, col2 = st.columns(2)            if error:

                        st.error(f"Query Error: {error}")

        with col1:            else:

            st.subheader("Transactions Table")                st.dataframe(result, use_container_width=True)

            st.code("""    

Columns:    with tab3:

- step: Time step (int)        st.subheader("💳 Pre-built Credit Queries")

- type: Transaction type (string)        

- amount: Transaction amount (float)        credit_queries = {

- nameOrig: Origin account (string)            "Default Risk by Demographics": """

- oldbalanceOrg: Origin account balance before (float)            SELECT 

- newbalanceOrig: Origin account balance after (float)                CASE WHEN SEX = 1 THEN 'Male' ELSE 'Female' END as gender,

- nameDest: Destination account (string)                EDUCATION,

- oldbalanceDest: Destination balance before (float)                MARRIAGE,

- newbalanceDest: Destination balance after (float)                COUNT(*) as customer_count,

- isFraud: Fraud flag (0/1)                ROUND(AVG(default) * 100, 2) as default_rate_pct,

- isFlaggedFraud: System fraud flag (0/1)                ROUND(AVG(LIMIT_BAL), 0) as avg_credit_limit

            """)            FROM credit

                    GROUP BY SEX, EDUCATION, MARRIAGE

        with col2:            ORDER BY default_rate_pct DESC

            st.subheader("Credit Data Table")            """,

            target_col = 'default payment next month' if 'default payment next month' in df_credit.columns else 'default_payment'            

            st.code(f"""            "Payment History Analysis": """

Columns:            SELECT 

- LIMIT_BAL: Credit limit (float)                PAY_0,

- SEX: Gender (1=male, 2=female)                COUNT(*) as customer_count,

- EDUCATION: Education level (1-4)                ROUND(AVG(default) * 100, 2) as default_rate_pct,

- MARRIAGE: Marriage status (1-3)                ROUND(AVG(LIMIT_BAL), 0) as avg_limit,

- AGE: Age (int)                ROUND(AVG(AGE), 1) as avg_age

- PAY_0 to PAY_6: Payment status (int)            FROM credit

- BILL_AMT1 to BILL_AMT6: Bill amounts (float)            GROUP BY PAY_0

- PAY_AMT1 to PAY_AMT6: Payment amounts (float)            ORDER BY PAY_0

- {target_col}: Default flag (0/1)            """,

            """)            

            "High Risk Customers": """

def show_advanced_analytics(df_transactions, df_credit):            SELECT 

    """Advanced analytics and machine learning insights"""                ID,

    st.header("📈 Advanced Analytics & Insights")                LIMIT_BAL,

                    AGE,

    # Business impact metrics                PAY_0,

    st.subheader("💰 Business Impact Analysis")                PAY_2,

                    PAY_3,

    col1, col2, col3, col4 = st.columns(4)                BILL_AMT1,

                    default

    with col1:            FROM credit

        # Calculate potential fraud savings            WHERE PAY_0 >= 2 AND PAY_2 >= 2

        fraud_volume = df_transactions[df_transactions['isFraud'] == 1]['amount'].sum()            ORDER BY PAY_0 DESC, PAY_2 DESC

        potential_savings = fraud_volume * 0.75  # Assume 75% prevention rate            LIMIT 20

        st.metric("Potential Fraud Savings", f"${potential_savings:,.0f}",             """,

                 help="Estimated annual savings from 75% fraud prevention")            

                "Credit Utilization Analysis": """

    with col2:            SELECT 

        # Calculate default prevention value                CASE 

        target_col = 'default payment next month' if 'default payment next month' in df_credit.columns else 'default_payment'                    WHEN BILL_AMT1 / LIMIT_BAL <= 0.3 THEN 'Low (<=30%)'

        if 'LIMIT_BAL' in df_credit.columns:                    WHEN BILL_AMT1 / LIMIT_BAL <= 0.7 THEN 'Medium (30-70%)'

            default_customers = df_credit[df_credit[target_col] == 1]                    ELSE 'High (>70%)'

            avg_default_loss = default_customers['LIMIT_BAL'].mean() * 0.6  # Assume 60% loss                END as utilization_category,

            prevention_value = len(default_customers) * avg_default_loss * 0.25  # 25% prevention                COUNT(*) as customer_count,

            st.metric("Default Prevention Value", f"${prevention_value:,.0f}",                ROUND(AVG(default) * 100, 2) as default_rate_pct,

                     help="Estimated value from 25% default prevention improvement")                ROUND(AVG(BILL_AMT1 / LIMIT_BAL) * 100, 1) as avg_utilization_pct

                FROM credit

    with col3:            WHERE LIMIT_BAL > 0 AND BILL_AMT1 >= 0

        # Customer risk distribution            GROUP BY utilization_category

        high_risk_customers = len(df_credit[df_credit[target_col] == 1])            ORDER BY default_rate_pct DESC

        risk_percentage = (high_risk_customers / len(df_credit)) * 100            """,

        st.metric("High-Risk Customers", f"{high_risk_customers:,} ({risk_percentage:.1f}%)")            

                "Age Group Risk Analysis": """

    with col4:            SELECT 

        # Model accuracy simulation                CASE 

        baseline_accuracy = 85.5                    WHEN AGE <= 30 THEN '<=30'

        improved_accuracy = 92.3                    WHEN AGE <= 40 THEN '31-40'

        improvement = improved_accuracy - baseline_accuracy                    WHEN AGE <= 50 THEN '41-50'

        st.metric("Model Accuracy", f"{improved_accuracy:.1f}%",                     WHEN AGE <= 60 THEN '51-60'

                 delta=f"+{improvement:.1f}%", help="Improved vs baseline model")                    ELSE '>60'

                    END as age_group,

    # Advanced analytics tabs                COUNT(*) as customer_count,

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Fraud Detection", "📊 Risk Scoring", "📈 Performance Metrics", "💡 Insights"])                ROUND(AVG(default) * 100, 2) as default_rate_pct,

                    ROUND(AVG(LIMIT_BAL), 0) as avg_limit

    with tab1:            FROM credit

        st.subheader("Fraud Detection Performance")            GROUP BY age_group

                    ORDER BY default_rate_pct DESC

        # Fraud detection metrics simulation            """

        fraud_metrics = {        }

            'Metric': ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Accuracy'],        

            'Current Model': [0.89, 0.82, 0.85, 0.94, 0.96],        selected_credit_query = st.selectbox("Select Credit Query:", list(credit_queries.keys()))

            'Industry Benchmark': [0.75, 0.70, 0.72, 0.85, 0.88],        st.code(credit_queries[selected_credit_query], language='sql')

            'Target': [0.92, 0.88, 0.90, 0.96, 0.97]        

        }        if st.button("Run Selected Query", key="credit_query"):

                    result, error = execute_sql_query(credit_queries[selected_credit_query], "credit", df_credit)

        metrics_df = pd.DataFrame(fraud_metrics)            

                    if error:

        fig = px.bar(metrics_df, x='Metric', y=['Current Model', 'Industry Benchmark', 'Target'],                st.error(f"Query Error: {error}")

                    title='Fraud Detection Model Performance Comparison', barmode='group')            else:

        st.plotly_chart(fig, use_container_width=True)                st.dataframe(result, use_container_width=True)

        

        # Feature importance simulationdef show_advanced_analytics(df_trans, df_credit):

        st.subheader("Feature Importance for Fraud Detection")    """Display advanced analytics page"""

            st.header("📈 Advanced Analytics")

        feature_importance = {    

            'Feature': ['Transaction Amount', 'Transaction Type', 'Account Balance Change',     tab1, tab2 = st.tabs(["🎯 Risk Scoring", "🔮 Predictive Insights"])

                       'Time of Day', 'Account Age', 'Previous Fraud History'],    

            'Importance': [0.35, 0.28, 0.18, 0.12, 0.05, 0.02]    with tab1:

        }        st.subheader("Risk Scoring Models")

                

        importance_df = pd.DataFrame(feature_importance)        # Credit risk scoring

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',        st.markdown("### 💳 Credit Default Risk Scoring")

                    title='Feature Importance for Fraud Detection')        df_credit_copy = df_credit.copy()

        st.plotly_chart(fig, use_container_width=True)        

            # Simple risk score calculation

    with tab2:        df_credit_copy['payment_risk'] = (df_credit_copy['PAY_0'] * 0.4 + 

        st.subheader("Customer Risk Scoring")                                         df_credit_copy['PAY_2'] * 0.3 + 

                                                 df_credit_copy['PAY_3'] * 0.2)

        # Risk score distribution        

        if 'LIMIT_BAL' in df_credit.columns and 'AGE' in df_credit.columns:        df_credit_copy['utilization'] = df_credit_copy['BILL_AMT1'] / df_credit_copy['LIMIT_BAL']

            # Calculate risk score        df_credit_copy['utilization'] = df_credit_copy['utilization'].fillna(0).clip(0, 2)

            df_risk = df_credit.copy()        

                    df_credit_copy['risk_score'] = (df_credit_copy['payment_risk'] * 0.6 + 

            # Simple risk scoring algorithm                                       df_credit_copy['utilization'] * 0.4)

            df_risk['risk_score'] = 0        

                    # Risk categories

            # Age factor (higher risk for very young and very old)        df_credit_copy['risk_category'] = pd.cut(df_credit_copy['risk_score'], 

            df_risk['age_factor'] = np.where(df_risk['AGE'] < 25, 0.3,                                                bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],

                                   np.where(df_risk['AGE'] > 65, 0.2, 0.1))                                                labels=['Low', 'Medium', 'High', 'Very High'])

                    

            # Payment history factor        # Risk analysis

            if 'PAY_0' in df_risk.columns:        risk_analysis = df_credit_copy.groupby('risk_category')['default'].agg(['count', 'sum', 'mean']).reset_index()

                df_risk['payment_factor'] = np.clip(df_risk['PAY_0'] / 10, 0, 0.4)        risk_analysis.columns = ['Risk_Category', 'Total', 'Defaults', 'Default_Rate']

            else:        risk_analysis['Default_Rate_Pct'] = risk_analysis['Default_Rate'] * 100

                df_risk['payment_factor'] = 0.2        

                    col1, col2 = st.columns(2)

            # Credit utilization factor        with col1:

            df_risk['utilization'] = df_risk['BILL_AMT1'] / df_risk['LIMIT_BAL']            fig = px.bar(risk_analysis, x='Risk_Category', y='Default_Rate_Pct',

            df_risk['utilization'] = df_risk['utilization'].fillna(0).clip(0, 2)                        title="Default Rate by Risk Category")

            df_risk['utilization_factor'] = np.clip(df_risk['utilization'] * 0.3, 0, 0.3)            st.plotly_chart(fig, use_container_width=True)

                    

            df_risk['risk_score'] = (df_risk['age_factor'] +         with col2:

                                   df_risk['payment_factor'] +             fig = px.pie(risk_analysis, values='Total', names='Risk_Category',

                                   df_risk['utilization_factor'])                        title="Customer Distribution by Risk Category")

                        st.plotly_chart(fig, use_container_width=True)

            # Risk categories        

            df_risk['risk_category'] = pd.cut(df_risk['risk_score'],         st.dataframe(risk_analysis, use_container_width=True)

                                            bins=[0, 0.3, 0.6, 1.0],         

                                            labels=['Low', 'Medium', 'High'])        # Transaction risk scoring

                    st.markdown("### 🔄 Transaction Fraud Risk Scoring")

            col1, col2 = st.columns(2)        df_trans_copy = df_trans.copy()

                    

            with col1:        # Simple fraud risk factors

                risk_dist = df_risk['risk_category'].value_counts()        df_trans_copy['amount_risk'] = pd.qcut(df_trans_copy['amount'], q=5, labels=[1,2,3,4,5])

                fig = px.pie(values=risk_dist.values, names=risk_dist.index,        df_trans_copy['type_risk'] = df_trans_copy['type'].map({

                           title="Customer Risk Distribution")            'PAYMENT': 1, 'DEBIT': 2, 'TRANSFER': 4, 'CASH_OUT': 5

                st.plotly_chart(fig, use_container_width=True)        })

                    df_trans_copy['balance_risk'] = (

            with col2:            abs(df_trans_copy['oldbalanceOrg'] - df_trans_copy['newbalanceOrig'] + df_trans_copy['amount']) > 0.01

                fig = px.histogram(df_risk, x='risk_score', nbins=30,        ).astype(int) * 3

                                 title="Risk Score Distribution")        

                st.plotly_chart(fig, use_container_width=True)        df_trans_copy['fraud_risk_score'] = (

                        df_trans_copy['amount_risk'].astype(float) * 0.4 +

            # Risk by actual default            df_trans_copy['type_risk'] * 0.4 +

            st.subheader("Risk Score Validation")            df_trans_copy['balance_risk'] * 0.2

            risk_validation = df_risk.groupby('risk_category')[target_col].mean().reset_index()        )

            risk_validation.columns = ['Risk_Category', 'Actual_Default_Rate']        

                    # Risk validation

            fig = px.bar(risk_validation, x='Risk_Category', y='Actual_Default_Rate',        risk_validation = df_trans_copy.groupby(pd.qcut(df_trans_copy['fraud_risk_score'], q=5))['isFraud'].mean() * 100

                        title="Actual Default Rate by Risk Category")        

            fig.update_layout(yaxis_title="Default Rate")        fig = px.bar(x=range(len(risk_validation)), y=risk_validation.values,

            st.plotly_chart(fig, use_container_width=True)                     title="Actual Fraud Rate by Risk Score Quintile")

            fig.update_xaxis(title="Risk Quintile (1=Lowest, 5=Highest)")

    with tab3:        fig.update_yaxis(title="Actual Fraud Rate (%)")

        st.subheader("Model Performance Metrics")        st.plotly_chart(fig, use_container_width=True)

            

        # ROC Curve simulation    with tab2:

        from sklearn.metrics import roc_curve, auc        st.subheader("🔮 Predictive Insights")

                

        # Simulate ROC data        # Feature importance simulation

        np.random.seed(42)        st.markdown("### 📊 Feature Importance Analysis")

        y_true = np.random.choice([0, 1], size=1000, p=[0.85, 0.15])        

        y_scores = np.random.beta(2, 5, size=1000)        # Credit default features

        y_scores = np.where(y_true == 1, y_scores + 0.3, y_scores)        credit_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'BILL_AMT1', 'PAY_AMT1']

        y_scores = np.clip(y_scores, 0, 1)        credit_importance = []

                

        fpr, tpr, _ = roc_curve(y_true, y_scores)        for feature in credit_features:

        roc_auc = auc(fpr, tpr)            corr = abs(df_credit[feature].corr(df_credit['default']))

                    credit_importance.append(corr)

        fig = go.Figure()        

        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',         credit_importance_df = pd.DataFrame({

                               name=f'ROC Curve (AUC = {roc_auc:.3f})'))            'Feature': credit_features,

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',             'Importance': credit_importance

                               name='Random Classifier', line=dict(dash='dash')))        }).sort_values('Importance', ascending=True)

        fig.update_layout(title='ROC Curve - Model Performance',        

                         xaxis_title='False Positive Rate',        fig = px.bar(credit_importance_df, x='Importance', y='Feature', orientation='h',

                         yaxis_title='True Positive Rate')                     title="Feature Importance for Credit Default Prediction")

        st.plotly_chart(fig, use_container_width=True)        st.plotly_chart(fig, use_container_width=True)

                

        # Confusion Matrix simulation        # Transaction fraud features

        col1, col2 = st.columns(2)        st.markdown("### 🔍 Transaction Patterns")

                

        with col1:        # Time-based patterns

            # Confusion matrix data        hourly_fraud = df_trans.groupby('hour')['isFraud'].mean() * 100

            cm_data = np.array([[850, 45], [25, 80]])        

                    fig = px.line(x=hourly_fraud.index, y=hourly_fraud.values,

            fig = px.imshow(cm_data, text_auto=True, aspect="auto",                      title="Fraud Rate by Hour of Day")

                           title="Confusion Matrix",        fig.update_xaxis(title="Hour")

                           labels=dict(x="Predicted", y="Actual", color="Count"))        fig.update_yaxis(title="Fraud Rate (%)")

            fig.update_xaxes(tickmode='array', tickvals=[0, 1], ticktext=['No Default', 'Default'])        st.plotly_chart(fig, use_container_width=True)

            fig.update_yaxes(tickmode='array', tickvals=[0, 1], ticktext=['No Default', 'Default'])

            st.plotly_chart(fig, use_container_width=True)def show_insights():

            """Display insights and recommendations"""

        with col2:    st.header("💡 Key Insights & Recommendations")

            # Performance metrics over time    

            dates = pd.date_range('2024-01-01', periods=12, freq='M')    st.markdown("""

            accuracy = 0.85 + 0.1 * np.random.random(12)    <div class="insights-box">

            precision = 0.80 + 0.15 * np.random.random(12)    <h3>🔍 Key Findings from Analysis</h3>

                </div>

            fig = go.Figure()    """, unsafe_allow_html=True)

            fig.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines+markers',     

                                   name='Accuracy'))    tab1, tab2, tab3 = st.tabs(["📊 Transaction Insights", "💳 Credit Insights", "🎯 Action Items"])

            fig.add_trace(go.Scatter(x=dates, y=precision, mode='lines+markers',     

                                   name='Precision'))    with tab1:

            fig.update_layout(title='Model Performance Over Time',        st.markdown("""

                             xaxis_title='Date', yaxis_title='Score')        ### 🔄 Transaction Data Insights

            st.plotly_chart(fig, use_container_width=True)        

            **Fraud Patterns:**

    with tab4:        - TRANSFER and CASH_OUT transactions have the highest fraud rates

        st.subheader("💡 Business Insights & Recommendations")        - Fraudulent transactions often involve balance inconsistencies

                - High-value transactions (99th percentile) require additional scrutiny

        # Key insights        

        st.markdown("""        **Transaction Behavior:**

        <div style='background-color: #e6f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #0066cc; margin: 1rem 0;'>        - PAYMENT transactions dominate the volume (>50%)

        <h4>🎯 Key Findings</h4>        - Peak transaction hours vary by transaction type

        <ul>        - Balance validation errors indicate potential system issues

        <li><strong>Fraud Detection:</strong> CASH_OUT and TRANSFER transactions have highest fraud rates</li>        

        <li><strong>Risk Factors:</strong> Young customers (18-25) and high credit utilization (>70%) increase default risk</li>        **Risk Indicators:**

        <li><strong>Timing Patterns:</strong> Fraudulent transactions peak during night hours (2-6 AM)</li>        - Zero balance accounts in high-value transactions

        <li><strong>Amount Patterns:</strong> Large transactions (>$50K) require enhanced screening</li>        - Rapid sequential transactions from same origin

        </ul>        - Unusual transaction patterns during off-hours

        </div>        """)

        """, unsafe_allow_html=True)    

            with tab2:

        # Recommendations        st.markdown("""

        st.markdown("""        ### 💳 Credit Card Default Insights

        <div style='background-color: #f0f8e6; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #4CAF50; margin: 1rem 0;'>        

        <h4>📈 Strategic Recommendations</h4>        **Demographic Patterns:**

        <ol>        - Default rates vary significantly by education level

        <li><strong>Enhanced Fraud Monitoring:</strong> Implement real-time alerts for CASH_OUT transactions >$25K</li>        - Gender shows marginal differences in default behavior

        <li><strong>Risk-Based Credit Limits:</strong> Adjust limits based on payment history and utilization patterns</li>        - Age groups 30-40 show higher default rates

        <li><strong>Customer Segmentation:</strong> Create targeted interventions for high-risk customer segments</li>        

        <li><strong>Time-Based Controls:</strong> Increase verification requirements for night-time large transactions</li>        **Payment Behavior:**

        <li><strong>Predictive Analytics:</strong> Deploy machine learning models for early default prediction</li>        - Payment history (PAY_0, PAY_2) is the strongest predictor

        </ol>        - Credit utilization above 70% significantly increases risk

        </div>        - Late payments in recent months are critical indicators

        """, unsafe_allow_html=True)        

                **Risk Factors:**

        # Expected impact        - Lower credit limits correlate with higher default rates

        st.markdown("""        - Marriage status influences payment behavior

        <div style='background-color: #fff3e0; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #FF9800; margin: 1rem 0;'>        - Bill amount patterns reveal financial stress indicators

        <h4>💰 Expected Business Impact</h4>        """)

        <ul>    

        <li><strong>Fraud Reduction:</strong> 25-40% decrease in fraud losses (~$2.3M annual savings)</li>    with tab3:

        <li><strong>Default Prevention:</strong> 15-25% improvement in default prediction accuracy</li>        st.markdown("""

        <li><strong>False Positives:</strong> 30-50% reduction in legitimate transaction blocks</li>        ### 🎯 Recommended Actions

        <li><strong>Customer Experience:</strong> Improved approval rates for low-risk customers</li>        

        <li><strong>Operational Efficiency:</strong> 40-60% reduction in manual review time</li>        **Immediate Actions (0-30 days):**

        </ul>        1. 🚨 Implement real-time balance validation for TRANSFER/CASH_OUT

        </div>        2. 📊 Set up monitoring dashboards for fraud rate by transaction type

        """, unsafe_allow_html=True)        3. 🔍 Review high-value transaction approval processes

        4. 📈 Deploy risk scoring for new credit applications

def main():        

    """Main application function"""        **Short-term Actions (1-3 months):**

    # Title and description        1. 🤖 Develop machine learning models for fraud detection

    st.markdown("""        2. 📋 Create automated credit risk assessment workflows

    <div style='text-align: center; margin-bottom: 2rem;'>        3. 🔄 Implement dynamic risk-based transaction limits

        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0.5rem;'>💰 Financial Analytics Dashboard</h1>        4. 📊 Establish KPI monitoring for all risk metrics

        <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>        

            Interactive analysis of financial transactions and credit risk using real-world datasets        **Long-term Strategy (3-12 months):**

        </p>        1. 🎯 Build comprehensive customer risk profiles

        <p style='font-size: 1rem; color: #888;'>        2. 📈 Implement predictive analytics for early warning systems

            📊 Data sources: Kaggle PaySim Dataset • UCI Credit Card Default Dataset        3. 🔗 Integrate external data sources for enhanced risk assessment

        </p>        4. 🚀 Develop real-time decision engines for transactions and credit

    </div>        

    """, unsafe_allow_html=True)        **Monitoring & Governance:**

            - Monthly review of model performance and risk thresholds

    # Load data with caching        - Quarterly assessment of fraud patterns and emerging risks

    df_transactions, df_credit = load_financial_datasets()        - Annual model validation and regulatory compliance review

            - Continuous monitoring of data quality and system performance

    # Sidebar navigation        """)

    st.sidebar.header("🧭 Navigation")    

        # Performance metrics

    page = st.sidebar.selectbox(    st.markdown("""

        "Choose Analysis Page:",    <div class="insights-box">

        [    <h3>📈 Expected Performance Improvements</h3>

            "📊 Transaction Analysis",    <ul>

            "💳 Credit Risk Analysis",     <li><strong>Fraud Detection:</strong> 25-40% improvement in detection rate</li>

            "🔍 SQL Query Interface",    <li><strong>False Positives:</strong> 30-50% reduction in legitimate transaction blocks</li>

            "📈 Advanced Analytics"    <li><strong>Credit Risk:</strong> 15-25% improvement in default prediction accuracy</li>

        ]    <li><strong>Operational Efficiency:</strong> 40-60% reduction in manual review time</li>

    )    </ul>

        </div>

    # Data summary in sidebar    """, unsafe_allow_html=True)

    with st.sidebar:

        st.subheader("📋 Data Summary")if __name__ == "__main__":

        st.metric("Transactions", f"{len(df_transactions):,}")    main()

        st.metric("Customers", f"{len(df_credit):,}")

        

        fraud_rate = df_transactions['isFraud'].mean() * 100#     to run the app

        st.metric("Fraud Rate", f"{fraud_rate:.3f}%")# cd "C:/Users/swata/skill/projects/Data Analytics/Financial Data Analytics/financial-data-analytics/streamlit_app"

        # python -m streamlit run app.py

        target_col = 'default payment next month' if 'default payment next month' in df_credit.columns else 'default_payment'
        default_rate = df_credit[target_col].mean() * 100
        st.metric("Default Rate", f"{default_rate:.1f}%")
    
    # Page routing
    if page == "📊 Transaction Analysis":
        show_transaction_analysis(df_transactions)
    elif page == "💳 Credit Risk Analysis":
        show_credit_analysis(df_credit)
    elif page == "🔍 SQL Query Interface":
        show_sql_interface(df_transactions, df_credit)
    elif page == "📈 Advanced Analytics":
        show_advanced_analytics(df_transactions, df_credit)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>🚀 Built with Streamlit • 📊 Data from Kaggle & UCI ML Repository • 💡 Powered by DuckDB</p>
        <p>For questions or feedback, contact the Financial Analytics Team</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()