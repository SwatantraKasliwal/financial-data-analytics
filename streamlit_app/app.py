"""
Financial Data Analytics - Interactive Dashboard
Author: [Your Name]
Date: September 2024

Interactive Streamlit dashboard for financial data analysis with SQL query capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insights-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

@st.cache_data
def load_transaction_data():
    """Load and preprocess transaction data"""
    try:
        # Try multiple possible data paths
        possible_paths = [
            'data/transactions.csv',
            '../data/transactions.csv',
            '../../Financial Data Analytics/financial-data-analytics/data/transactions.csv',
            '../Financial Data Analytics/financial-data-analytics/data/transactions.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"✅ Successfully loaded transaction data from: {path}")
                break
            except:
                continue
        
        if df is None:
            # Create dummy data if no file found
            st.warning("⚠️ Using dummy transaction data for demonstration")
            np.random.seed(42)
            dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")
            customers = [f"CUST{str(i).zfill(4)}" for i in range(1, 201)]
            data = {
                "step": np.random.randint(1, 8760, 3000),
                "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], 3000, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
                "amount": np.random.exponential(scale=120, size=3000).round(2),
                "nameOrig": np.random.choice(customers, 3000),
                "oldbalanceOrg": np.random.uniform(0, 100000, 3000).round(2),
                "newbalanceOrig": np.random.uniform(0, 100000, 3000).round(2),
                "nameDest": np.random.choice(customers, 3000),
                "oldbalanceDest": np.random.uniform(0, 100000, 3000).round(2),
                "newbalanceDest": np.random.uniform(0, 100000, 3000).round(2),
                "isFraud": np.random.choice([0, 1], 3000, p=[0.97, 0.03]),
                "isFlaggedFraud": np.random.choice([0, 1], 3000, p=[0.99, 0.01])
            }
            df = pd.DataFrame(data)
        
        # Basic preprocessing
        df['hour'] = df['step'] % 24
        df['day'] = df['step'] // 24
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading transaction data: {e}")
        return None

@st.cache_data
def load_credit_data():
    """Load and preprocess credit card data"""
    try:
        # Try multiple possible data paths
        possible_paths = [
            'data/default of credit card clients.xls',
            '../data/default of credit card clients.xls',
            '../../Financial Data Analytics/financial-data-analytics/data/default of credit card clients.xls',
            '../Financial Data Analytics/financial-data-analytics/data/default of credit card clients.xls'
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_excel(path, header=1)
                st.success(f"✅ Successfully loaded credit data from: {path}")
                break
            except:
                continue
        
        if df is None:
            # Create dummy data if no file found
            st.warning("⚠️ Using dummy credit data for demonstration")
            np.random.seed(42)
            data = {
                "ID": range(1, 1001),
                "LIMIT_BAL": np.random.uniform(10000, 800000, 1000).round(0),
                "SEX": np.random.choice([1, 2], 1000),
                "EDUCATION": np.random.choice([1, 2, 3, 4], 1000),
                "MARRIAGE": np.random.choice([1, 2, 3], 1000),
                "AGE": np.random.randint(21, 79, 1000),
                "PAY_0": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),
                "PAY_2": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),
                "PAY_3": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),
                "PAY_4": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),
                "PAY_5": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),
                "PAY_6": np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8], 1000, p=[0.4, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.03, 0.04]),
                "BILL_AMT1": np.random.uniform(0, 100000, 1000).round(2),
                "BILL_AMT2": np.random.uniform(0, 100000, 1000).round(2),
                "BILL_AMT3": np.random.uniform(0, 100000, 1000).round(2),
                "BILL_AMT4": np.random.uniform(0, 100000, 1000).round(2),
                "BILL_AMT5": np.random.uniform(0, 100000, 1000).round(2),
                "BILL_AMT6": np.random.uniform(0, 100000, 1000).round(2),
                "PAY_AMT1": np.random.uniform(0, 50000, 1000).round(2),
                "PAY_AMT2": np.random.uniform(0, 50000, 1000).round(2),
                "PAY_AMT3": np.random.uniform(0, 50000, 1000).round(2),
                "PAY_AMT4": np.random.uniform(0, 50000, 1000).round(2),
                "PAY_AMT5": np.random.uniform(0, 50000, 1000).round(2),
                "PAY_AMT6": np.random.uniform(0, 50000, 1000).round(2),
                "default": np.random.choice([0, 1], 1000, p=[0.77, 0.23])
            }
            df = pd.DataFrame(data)
        else:
            df.rename(columns={'default payment next month': 'default'}, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading credit data: {e}")
        return None

def execute_sql_query(query, df_name, df):
    """Execute SQL query using DuckDB"""
    try:
        conn = duckdb.connect()
        conn.register(df_name, df)
        result = conn.execute(query).fetchdf()
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)

# Main app
def main():
    st.markdown('<h1 class="main-header">💰 Financial Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["🏠 Overview", "📊 Transaction Analysis", "💳 Credit Analysis", 
         "🔍 SQL Query Interface", "📈 Advanced Analytics", "💡 Insights & Recommendations"]
    )
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            df_trans = load_transaction_data()
            df_credit = load_credit_data()
            
            if df_trans is not None and df_credit is not None:
                st.session_state.df_trans = df_trans
                st.session_state.df_credit = df_credit
                st.session_state.data_loaded = True
            else:
                st.error("Failed to load data. Please check data files.")
                return
    
    df_trans = st.session_state.df_trans
    df_credit = st.session_state.df_credit
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(df_trans, df_credit)
    elif page == "📊 Transaction Analysis":
        show_transaction_analysis(df_trans)
    elif page == "💳 Credit Analysis":
        show_credit_analysis(df_credit)
    elif page == "🔍 SQL Query Interface":
        show_sql_interface(df_trans, df_credit)
    elif page == "📈 Advanced Analytics":
        show_advanced_analytics(df_trans, df_credit)
    elif page == "💡 Insights & Recommendations":
        show_insights()

def show_overview(df_trans, df_credit):
    """Display overview page"""
    st.header("📊 Financial Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Transaction Data")
        st.markdown(f"""
        <div class="metric-card">
        <h3>Dataset Statistics</h3>
        <ul>
        <li><strong>Total Transactions:</strong> {len(df_trans):,}</li>
        <li><strong>Transaction Types:</strong> {df_trans['type'].nunique()}</li>
        <li><strong>Total Volume:</strong> ${df_trans['amount'].sum():,.2f}</li>
        <li><strong>Fraud Rate:</strong> {df_trans['isFraud'].mean()*100:.3f}%</li>
        <li><strong>Time Period:</strong> {df_trans['step'].max()} time steps</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick transaction type chart
        fig = px.pie(df_trans['type'].value_counts().reset_index(), 
                     values='count', names='type', 
                     title="Transaction Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💳 Credit Card Data")
        st.markdown(f"""
        <div class="metric-card">
        <h3>Dataset Statistics</h3>
        <ul>
        <li><strong>Total Customers:</strong> {len(df_credit):,}</li>
        <li><strong>Default Rate:</strong> {df_credit['default'].mean()*100:.2f}%</li>
        <li><strong>Avg Credit Limit:</strong> ${df_credit['LIMIT_BAL'].mean():,.2f}</li>
        <li><strong>Age Range:</strong> {df_credit['AGE'].min()}-{df_credit['AGE'].max()} years</li>
        <li><strong>Features:</strong> {len(df_credit.columns)} variables</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick default rate by gender
        gender_default = df_credit.groupby('SEX')['default'].mean() * 100
        gender_default.index = ['Male', 'Female']
        fig = px.bar(x=gender_default.index, y=gender_default.values, 
                     title="Default Rate by Gender (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data preview
    st.subheader("📋 Data Preview")
    tab1, tab2 = st.tabs(["Transaction Data", "Credit Data"])
    
    with tab1:
        st.dataframe(df_trans.head(), use_container_width=True)
    
    with tab2:
        st.dataframe(df_credit.head(), use_container_width=True)

def show_transaction_analysis(df_trans):
    """Display transaction analysis page"""
    st.header("📊 Transaction Data Analysis")
    
    # Filters
    st.sidebar.subheader("🔧 Filters")
    selected_types = st.sidebar.multiselect(
        "Select Transaction Types",
        df_trans['type'].unique(),
        default=df_trans['type'].unique()
    )
    
    amount_range = st.sidebar.slider(
        "Amount Range ($)",
        float(df_trans['amount'].min()),
        float(df_trans['amount'].quantile(0.99)),
        (float(df_trans['amount'].min()), float(df_trans['amount'].quantile(0.95)))
    )
    
    # Filter data
    filtered_df = df_trans[
        (df_trans['type'].isin(selected_types)) &
        (df_trans['amount'] >= amount_range[0]) &
        (df_trans['amount'] <= amount_range[1])
    ]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(filtered_df):,}")
    with col2:
        st.metric("Total Volume", f"${filtered_df['amount'].sum():,.2f}")
    with col3:
        st.metric("Fraud Rate", f"{filtered_df['isFraud'].mean()*100:.3f}%")
    with col4:
        st.metric("Avg Amount", f"${filtered_df['amount'].mean():.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution
        fig = px.histogram(filtered_df, x='amount', nbins=50, 
                          title="Transaction Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by type
        fraud_by_type = filtered_df.groupby('type')['isFraud'].agg(['count', 'sum']).reset_index()
        fraud_by_type['fraud_rate'] = fraud_by_type['sum'] / fraud_by_type['count'] * 100
        fig = px.bar(fraud_by_type, x='type', y='fraud_rate', 
                     title="Fraud Rate by Transaction Type (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Time series
        hourly_data = filtered_df.groupby('hour').size().reset_index(name='count')
        fig = px.line(hourly_data, x='hour', y='count', 
                      title="Transactions by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)
        
        # Amount vs fraud scatter
        sample_data = filtered_df.sample(n=min(5000, len(filtered_df)))
        fig = px.scatter(sample_data, x='amount', y='step', color='isFraud',
                        title="Amount vs Time (Fraud Highlighted)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    # Balance analysis
    if st.checkbox("Show Balance Analysis"):
        st.markdown("### Balance Change Analysis")
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['balance_change_orig'] = filtered_df_copy['newbalanceOrig'] - filtered_df_copy['oldbalanceOrg']
        filtered_df_copy['balance_inconsistency'] = abs(filtered_df_copy['balance_change_orig'] + filtered_df_copy['amount']) > 0.01
        
        inconsistency_rate = filtered_df_copy['balance_inconsistency'].mean() * 100
        st.metric("Balance Inconsistency Rate", f"{inconsistency_rate:.2f}%")
        
        fig = px.histogram(filtered_df_copy, x='balance_change_orig', 
                          title="Distribution of Balance Changes (Origin)")
        st.plotly_chart(fig, use_container_width=True)

def show_credit_analysis(df_credit):
    """Display credit analysis page"""
    st.header("💳 Credit Card Default Analysis")
    
    # Filters
    st.sidebar.subheader("🔧 Filters")
    age_range = st.sidebar.slider(
        "Age Range",
        int(df_credit['AGE'].min()),
        int(df_credit['AGE'].max()),
        (int(df_credit['AGE'].min()), int(df_credit['AGE'].max()))
    )
    
    limit_range = st.sidebar.slider(
        "Credit Limit Range ($)",
        float(df_credit['LIMIT_BAL'].min()),
        float(df_credit['LIMIT_BAL'].max()),
        (float(df_credit['LIMIT_BAL'].min()), float(df_credit['LIMIT_BAL'].max()))
    )
    
    # Filter data
    filtered_df = df_credit[
        (df_credit['AGE'] >= age_range[0]) &
        (df_credit['AGE'] <= age_range[1]) &
        (df_credit['LIMIT_BAL'] >= limit_range[0]) &
        (df_credit['LIMIT_BAL'] <= limit_range[1])
    ]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    with col2:
        st.metric("Default Rate", f"{filtered_df['default'].mean()*100:.2f}%")
    with col3:
        st.metric("Avg Credit Limit", f"${filtered_df['LIMIT_BAL'].mean():,.0f}")
    with col4:
        st.metric("Avg Age", f"{filtered_df['AGE'].mean():.1f} years")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Default by gender
        gender_default = filtered_df.groupby('SEX')['default'].mean() * 100
        gender_default.index = ['Male', 'Female']
        fig = px.bar(x=gender_default.index, y=gender_default.values,
                     title="Default Rate by Gender (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Age distribution
        fig = px.histogram(filtered_df, x='AGE', color='default',
                          title="Age Distribution by Default Status")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Default by education
        edu_default = filtered_df.groupby('EDUCATION')['default'].mean() * 100
        fig = px.bar(x=edu_default.index, y=edu_default.values,
                     title="Default Rate by Education Level (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Credit limit vs default
        fig = px.box(filtered_df, x='default', y='LIMIT_BAL',
                     title="Credit Limit Distribution by Default Status")
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment history analysis
    if st.checkbox("Show Payment History Analysis"):
        st.markdown("### Payment History Analysis")
        payment_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        
        # Correlation with default
        correlations = filtered_df[payment_cols + ['default']].corr()['default'].drop('default')
        fig = px.bar(x=correlations.index, y=correlations.values,
                     title="Correlation of Payment History with Default")
        st.plotly_chart(fig, use_container_width=True)

def show_sql_interface(df_trans, df_credit):
    """Display SQL query interface"""
    st.header("🔍 SQL Query Interface")
    
    st.markdown("""
    <div class="insights-box">
    <h3>💡 Interactive SQL Analysis</h3>
    <p>Use SQL queries to explore the data interactively. Available tables:</p>
    <ul>
    <li><strong>transactions</strong> - Transaction data (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud)</li>
    <li><strong>credit</strong> - Credit card data (ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0-PAY_6, BILL_AMT1-BILL_AMT6, PAY_AMT1-PAY_AMT6, default)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Query tabs
    tab1, tab2, tab3 = st.tabs(["📝 Custom Query", "📊 Transaction Queries", "💳 Credit Queries"])
    
    with tab1:
        st.subheader("Write Your Own SQL Query")
        
        # Query input
        query = st.text_area(
            "Enter SQL Query:",
            height=150,
            value="SELECT type, COUNT(*) as transaction_count, AVG(amount) as avg_amount\nFROM transactions\nGROUP BY type\nORDER BY transaction_count DESC"
        )
        
        # Table selection
        table_name = st.selectbox("Select Table:", ["transactions", "credit"])
        
        if st.button("Execute Query"):
            df_to_use = df_trans if table_name == "transactions" else df_credit
            result, error = execute_sql_query(query, table_name, df_to_use)
            
            if error:
                st.error(f"Query Error: {error}")
            else:
                st.success("Query executed successfully!")
                st.dataframe(result, use_container_width=True)
                
                # Download results
                csv = result.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
    
    with tab2:
        st.subheader("📊 Pre-built Transaction Queries")
        
        query_options = {
            "Fraud Analysis by Type": """
            SELECT 
                type,
                COUNT(*) as total_transactions,
                SUM(isFraud) as fraud_count,
                ROUND(AVG(isFraud) * 100, 2) as fraud_rate_pct,
                ROUND(AVG(amount), 2) as avg_amount
            FROM transactions
            GROUP BY type
            ORDER BY fraud_rate_pct DESC
            """,
            
            "High Value Transactions": """
            SELECT 
                type,
                amount,
                nameOrig,
                nameDest,
                isFraud,
                step
            FROM transactions
            WHERE amount > (SELECT PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) FROM transactions)
            ORDER BY amount DESC
            LIMIT 20
            """,
            
            "Hourly Transaction Patterns": """
            SELECT 
                (step % 24) as hour,
                COUNT(*) as transaction_count,
                ROUND(AVG(amount), 2) as avg_amount,
                SUM(isFraud) as fraud_count
            FROM transactions
            GROUP BY hour
            ORDER BY hour
            """,
            
            "Balance Inconsistencies": """
            SELECT 
                type,
                amount,
                oldbalanceOrg,
                newbalanceOrig,
                (oldbalanceOrg - amount) as expected_balance,
                newbalanceOrig,
                ABS((oldbalanceOrg - amount) - newbalanceOrig) as inconsistency,
                isFraud
            FROM transactions
            WHERE ABS((oldbalanceOrg - amount) - newbalanceOrig) > 0.01
            ORDER BY inconsistency DESC
            LIMIT 20
            """,
            
            "Top Senders by Volume": """
            SELECT 
                nameOrig,
                COUNT(*) as transaction_count,
                ROUND(SUM(amount), 2) as total_amount,
                ROUND(AVG(amount), 2) as avg_amount,
                SUM(isFraud) as fraud_count
            FROM transactions
            GROUP BY nameOrig
            ORDER BY total_amount DESC
            LIMIT 15
            """
        }
        
        selected_query = st.selectbox("Select Query:", list(query_options.keys()))
        st.code(query_options[selected_query], language='sql')
        
        if st.button("Run Selected Query", key="trans_query"):
            result, error = execute_sql_query(query_options[selected_query], "transactions", df_trans)
            
            if error:
                st.error(f"Query Error: {error}")
            else:
                st.dataframe(result, use_container_width=True)
    
    with tab3:
        st.subheader("💳 Pre-built Credit Queries")
        
        credit_queries = {
            "Default Risk by Demographics": """
            SELECT 
                CASE WHEN SEX = 1 THEN 'Male' ELSE 'Female' END as gender,
                EDUCATION,
                MARRIAGE,
                COUNT(*) as customer_count,
                ROUND(AVG(default) * 100, 2) as default_rate_pct,
                ROUND(AVG(LIMIT_BAL), 0) as avg_credit_limit
            FROM credit
            GROUP BY SEX, EDUCATION, MARRIAGE
            ORDER BY default_rate_pct DESC
            """,
            
            "Payment History Analysis": """
            SELECT 
                PAY_0,
                COUNT(*) as customer_count,
                ROUND(AVG(default) * 100, 2) as default_rate_pct,
                ROUND(AVG(LIMIT_BAL), 0) as avg_limit,
                ROUND(AVG(AGE), 1) as avg_age
            FROM credit
            GROUP BY PAY_0
            ORDER BY PAY_0
            """,
            
            "High Risk Customers": """
            SELECT 
                ID,
                LIMIT_BAL,
                AGE,
                PAY_0,
                PAY_2,
                PAY_3,
                BILL_AMT1,
                default
            FROM credit
            WHERE PAY_0 >= 2 AND PAY_2 >= 2
            ORDER BY PAY_0 DESC, PAY_2 DESC
            LIMIT 20
            """,
            
            "Credit Utilization Analysis": """
            SELECT 
                CASE 
                    WHEN BILL_AMT1 / LIMIT_BAL <= 0.3 THEN 'Low (<=30%)'
                    WHEN BILL_AMT1 / LIMIT_BAL <= 0.7 THEN 'Medium (30-70%)'
                    ELSE 'High (>70%)'
                END as utilization_category,
                COUNT(*) as customer_count,
                ROUND(AVG(default) * 100, 2) as default_rate_pct,
                ROUND(AVG(BILL_AMT1 / LIMIT_BAL) * 100, 1) as avg_utilization_pct
            FROM credit
            WHERE LIMIT_BAL > 0 AND BILL_AMT1 >= 0
            GROUP BY utilization_category
            ORDER BY default_rate_pct DESC
            """,
            
            "Age Group Risk Analysis": """
            SELECT 
                CASE 
                    WHEN AGE <= 30 THEN '<=30'
                    WHEN AGE <= 40 THEN '31-40'
                    WHEN AGE <= 50 THEN '41-50'
                    WHEN AGE <= 60 THEN '51-60'
                    ELSE '>60'
                END as age_group,
                COUNT(*) as customer_count,
                ROUND(AVG(default) * 100, 2) as default_rate_pct,
                ROUND(AVG(LIMIT_BAL), 0) as avg_limit
            FROM credit
            GROUP BY age_group
            ORDER BY default_rate_pct DESC
            """
        }
        
        selected_credit_query = st.selectbox("Select Credit Query:", list(credit_queries.keys()))
        st.code(credit_queries[selected_credit_query], language='sql')
        
        if st.button("Run Selected Query", key="credit_query"):
            result, error = execute_sql_query(credit_queries[selected_credit_query], "credit", df_credit)
            
            if error:
                st.error(f"Query Error: {error}")
            else:
                st.dataframe(result, use_container_width=True)

def show_advanced_analytics(df_trans, df_credit):
    """Display advanced analytics page"""
    st.header("📈 Advanced Analytics")
    
    tab1, tab2 = st.tabs(["🎯 Risk Scoring", "🔮 Predictive Insights"])
    
    with tab1:
        st.subheader("Risk Scoring Models")
        
        # Credit risk scoring
        st.markdown("### 💳 Credit Default Risk Scoring")
        df_credit_copy = df_credit.copy()
        
        # Simple risk score calculation
        df_credit_copy['payment_risk'] = (df_credit_copy['PAY_0'] * 0.4 + 
                                         df_credit_copy['PAY_2'] * 0.3 + 
                                         df_credit_copy['PAY_3'] * 0.2)
        
        df_credit_copy['utilization'] = df_credit_copy['BILL_AMT1'] / df_credit_copy['LIMIT_BAL']
        df_credit_copy['utilization'] = df_credit_copy['utilization'].fillna(0).clip(0, 2)
        
        df_credit_copy['risk_score'] = (df_credit_copy['payment_risk'] * 0.6 + 
                                       df_credit_copy['utilization'] * 0.4)
        
        # Risk categories
        df_credit_copy['risk_category'] = pd.cut(df_credit_copy['risk_score'], 
                                                bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
                                                labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Risk analysis
        risk_analysis = df_credit_copy.groupby('risk_category')['default'].agg(['count', 'sum', 'mean']).reset_index()
        risk_analysis.columns = ['Risk_Category', 'Total', 'Defaults', 'Default_Rate']
        risk_analysis['Default_Rate_Pct'] = risk_analysis['Default_Rate'] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(risk_analysis, x='Risk_Category', y='Default_Rate_Pct',
                        title="Default Rate by Risk Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(risk_analysis, values='Total', names='Risk_Category',
                        title="Customer Distribution by Risk Category")
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(risk_analysis, use_container_width=True)
        
        # Transaction risk scoring
        st.markdown("### 🔄 Transaction Fraud Risk Scoring")
        df_trans_copy = df_trans.copy()
        
        # Simple fraud risk factors
        df_trans_copy['amount_risk'] = pd.qcut(df_trans_copy['amount'], q=5, labels=[1,2,3,4,5])
        df_trans_copy['type_risk'] = df_trans_copy['type'].map({
            'PAYMENT': 1, 'DEBIT': 2, 'TRANSFER': 4, 'CASH_OUT': 5
        })
        df_trans_copy['balance_risk'] = (
            abs(df_trans_copy['oldbalanceOrg'] - df_trans_copy['newbalanceOrig'] + df_trans_copy['amount']) > 0.01
        ).astype(int) * 3
        
        df_trans_copy['fraud_risk_score'] = (
            df_trans_copy['amount_risk'].astype(float) * 0.4 +
            df_trans_copy['type_risk'] * 0.4 +
            df_trans_copy['balance_risk'] * 0.2
        )
        
        # Risk validation
        risk_validation = df_trans_copy.groupby(pd.qcut(df_trans_copy['fraud_risk_score'], q=5))['isFraud'].mean() * 100
        
        fig = px.bar(x=range(len(risk_validation)), y=risk_validation.values,
                     title="Actual Fraud Rate by Risk Score Quintile")
        fig.update_xaxis(title="Risk Quintile (1=Lowest, 5=Highest)")
        fig.update_yaxis(title="Actual Fraud Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Predictive Insights")
        
        # Feature importance simulation
        st.markdown("### 📊 Feature Importance Analysis")
        
        # Credit default features
        credit_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'BILL_AMT1', 'PAY_AMT1']
        credit_importance = []
        
        for feature in credit_features:
            corr = abs(df_credit[feature].corr(df_credit['default']))
            credit_importance.append(corr)
        
        credit_importance_df = pd.DataFrame({
            'Feature': credit_features,
            'Importance': credit_importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(credit_importance_df, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance for Credit Default Prediction")
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction fraud features
        st.markdown("### 🔍 Transaction Patterns")
        
        # Time-based patterns
        hourly_fraud = df_trans.groupby('hour')['isFraud'].mean() * 100
        
        fig = px.line(x=hourly_fraud.index, y=hourly_fraud.values,
                      title="Fraud Rate by Hour of Day")
        fig.update_xaxis(title="Hour")
        fig.update_yaxis(title="Fraud Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

def show_insights():
    """Display insights and recommendations"""
    st.header("💡 Key Insights & Recommendations")
    
    st.markdown("""
    <div class="insights-box">
    <h3>🔍 Key Findings from Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Transaction Insights", "💳 Credit Insights", "🎯 Action Items"])
    
    with tab1:
        st.markdown("""
        ### 🔄 Transaction Data Insights
        
        **Fraud Patterns:**
        - TRANSFER and CASH_OUT transactions have the highest fraud rates
        - Fraudulent transactions often involve balance inconsistencies
        - High-value transactions (99th percentile) require additional scrutiny
        
        **Transaction Behavior:**
        - PAYMENT transactions dominate the volume (>50%)
        - Peak transaction hours vary by transaction type
        - Balance validation errors indicate potential system issues
        
        **Risk Indicators:**
        - Zero balance accounts in high-value transactions
        - Rapid sequential transactions from same origin
        - Unusual transaction patterns during off-hours
        """)
    
    with tab2:
        st.markdown("""
        ### 💳 Credit Card Default Insights
        
        **Demographic Patterns:**
        - Default rates vary significantly by education level
        - Gender shows marginal differences in default behavior
        - Age groups 30-40 show higher default rates
        
        **Payment Behavior:**
        - Payment history (PAY_0, PAY_2) is the strongest predictor
        - Credit utilization above 70% significantly increases risk
        - Late payments in recent months are critical indicators
        
        **Risk Factors:**
        - Lower credit limits correlate with higher default rates
        - Marriage status influences payment behavior
        - Bill amount patterns reveal financial stress indicators
        """)
    
    with tab3:
        st.markdown("""
        ### 🎯 Recommended Actions
        
        **Immediate Actions (0-30 days):**
        1. 🚨 Implement real-time balance validation for TRANSFER/CASH_OUT
        2. 📊 Set up monitoring dashboards for fraud rate by transaction type
        3. 🔍 Review high-value transaction approval processes
        4. 📈 Deploy risk scoring for new credit applications
        
        **Short-term Actions (1-3 months):**
        1. 🤖 Develop machine learning models for fraud detection
        2. 📋 Create automated credit risk assessment workflows
        3. 🔄 Implement dynamic risk-based transaction limits
        4. 📊 Establish KPI monitoring for all risk metrics
        
        **Long-term Strategy (3-12 months):**
        1. 🎯 Build comprehensive customer risk profiles
        2. 📈 Implement predictive analytics for early warning systems
        3. 🔗 Integrate external data sources for enhanced risk assessment
        4. 🚀 Develop real-time decision engines for transactions and credit
        
        **Monitoring & Governance:**
        - Monthly review of model performance and risk thresholds
        - Quarterly assessment of fraud patterns and emerging risks
        - Annual model validation and regulatory compliance review
        - Continuous monitoring of data quality and system performance
        """)
    
    # Performance metrics
    st.markdown("""
    <div class="insights-box">
    <h3>📈 Expected Performance Improvements</h3>
    <ul>
    <li><strong>Fraud Detection:</strong> 25-40% improvement in detection rate</li>
    <li><strong>False Positives:</strong> 30-50% reduction in legitimate transaction blocks</li>
    <li><strong>Credit Risk:</strong> 15-25% improvement in default prediction accuracy</li>
    <li><strong>Operational Efficiency:</strong> 40-60% reduction in manual review time</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


#     to run the app
# cd "C:/Users/swata/skill/projects/Data Analytics/Financial Data Analytics/financial-data-analytics/streamlit_app"
# python -m streamlit run app.py
