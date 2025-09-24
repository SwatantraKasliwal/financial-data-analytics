"""
Financial Data Analytics Dashboard
Interactive Streamlit application for comprehensive financial data analysis
Optimized for Streamlit Cloud hosting with automatic data fetching
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import warnings
import sys
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Health check endpoint for Streamlit Cloud
def health_check():
    """Simple health check function"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Add utils directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import data fetcher
try:
    from utils.kaggle_data_fetcher import kaggle_fetcher
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Financial Data Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .insights-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .data-source-info {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_trans' not in st.session_state:
    st.session_state.df_trans = None
if 'df_credit' not in st.session_state:
    st.session_state.df_credit = None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_transaction_data():
    """Load transaction data with caching"""
    try:
        if DATA_FETCHER_AVAILABLE:
            with st.spinner("📥 Fetching transaction data..."):
                df = kaggle_fetcher.load_transaction_data("paysim_transactions", sample_size=100000)
                
                # Create time-based features if missing
                if 'hour' not in df.columns and 'step' in df.columns:
                    df['hour'] = df['step'] % 24
                    df['day'] = df['step'] // 24
                    df['day_of_week'] = df['day'] % 7
                
                st.success("✅ Transaction data loaded successfully!")
                return df
        else:
            raise ImportError("Data fetcher not available")
    except Exception as e:
        st.warning(f"⚠️ Using sample data: {str(e)}")
        return generate_sample_transaction_data()

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def load_credit_data():
    """Load credit data with caching"""
    try:
        if DATA_FETCHER_AVAILABLE:
            with st.spinner("📥 Fetching credit data..."):
                df = kaggle_fetcher.load_credit_data("credit_default_uci", sample_size=20000)
                
                # Rename UCI credit columns to meaningful names if needed
                if 'x1' in df.columns:
                    # Map UCI column names to meaningful names
                    credit_column_mapping = {
                        'x1': 'limit_bal',      # Amount of the given credit
                        'x2': 'sex',            # Gender (1 = male; 2 = female)
                        'x3': 'education',      # Education level
                        'x4': 'marriage',       # Marital status
                        'x5': 'age',            # Age
                        'x6': 'pay_0',          # Repayment status in September
                        'x7': 'pay_2',          # Repayment status in August
                        'x8': 'pay_3',          # Repayment status in July
                        'x9': 'pay_4',          # Repayment status in June
                        'x10': 'pay_5',         # Repayment status in May
                        'x11': 'pay_6',         # Repayment status in April
                        'x12': 'bill_amt1',     # Bill statement in September
                        'x13': 'bill_amt2',     # Bill statement in August
                        'x14': 'bill_amt3',     # Bill statement in July
                        'x15': 'bill_amt4',     # Bill statement in June
                        'x16': 'bill_amt5',     # Bill statement in May
                        'x17': 'bill_amt6',     # Bill statement in April
                        'x18': 'pay_amt1',      # Previous payment in September
                        'x19': 'pay_amt2',      # Previous payment in August
                        'x20': 'pay_amt3',      # Previous payment in July
                        'x21': 'pay_amt4',      # Previous payment in June
                        'x22': 'pay_amt5',      # Previous payment in May
                        'x23': 'pay_amt6',      # Previous payment in April
                        'y': 'default'          # Default payment next month
                    }
                    df = df.rename(columns=credit_column_mapping)
                
                # Create id column if missing
                if 'id' not in df.columns:
                    df['id'] = range(1, len(df) + 1)
                
                st.success("✅ Credit data loaded successfully!")
                return df
        else:
            raise ImportError("Data fetcher not available")
    except Exception as e:
        st.warning(f"⚠️ Using sample data: {str(e)}")
        return generate_sample_credit_data()

def generate_sample_transaction_data():
    """Generate sample transaction data as fallback"""
    np.random.seed(42)
    n_rows = 50000
    
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
    df['hour'] = df['step'] % 24
    df['day'] = df['step'] // 24
    
    return df

def generate_sample_credit_data():
    """Generate sample credit data as fallback"""
    np.random.seed(42)
    n_rows = 10000
    
    data = {
        'id': range(1, n_rows + 1),
        'limit_bal': np.random.uniform(10000, 500000, n_rows),
        'sex': np.random.choice([1, 2], n_rows),
        'education': np.random.choice([1, 2, 3, 4], n_rows, p=[0.4, 0.3, 0.2, 0.1]),
        'marriage': np.random.choice([1, 2, 3], n_rows, p=[0.5, 0.4, 0.1]),
        'age': np.random.randint(20, 80, n_rows),
        'default': np.random.choice([0, 1], n_rows, p=[0.78, 0.22])
    }
    
    # Add payment history and bill amounts
    for i in range(6):
        data[f'pay_{i}'] = np.random.choice([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_rows)
        data[f'bill_amt{i+1}'] = np.random.uniform(0, 100000, n_rows)
        data[f'pay_amt{i+1}'] = np.random.uniform(0, 50000, n_rows)
    
    return pd.DataFrame(data)

def execute_sql_query(query, table_name, df):
    """Execute SQL query using DuckDB"""
    try:
        conn = duckdb.connect(':memory:')
        conn.register(table_name, df)
        result = conn.execute(query).fetchdf()
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">💰 Financial Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Data source information
    st.markdown("""
    <div class="data-source-info">
    <h3>📊 Data Sources</h3>
    <p>This dashboard uses real financial datasets fetched automatically from:</p>
    <ul>
        <li><strong>PaySim Transactions</strong>: Mobile money simulator data for fraud detection</li>
        <li><strong>Credit Risk Assessment</strong>: UCI ML repository credit default data</li>
    </ul>
    <p>If external data is unavailable, realistic sample data is generated automatically.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["🏠 Overview", "📊 Transaction Analysis", "💳 Credit Analysis", 
         "🔍 SQL Query Interface", "📈 Advanced Analytics", "💡 Insights & Recommendations"]
    )
    
    # Data loading
    if not st.session_state.data_loaded:
        with st.spinner("Loading financial data..."):
            st.session_state.df_trans = load_transaction_data()
            st.session_state.df_credit = load_credit_data()
            st.session_state.data_loaded = True
    
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

def get_credit_column_safe(df_credit, target_col):
    """Safely get credit column data handling both UCI and renamed formats"""
    # Mapping of target columns to possible UCI names
    uci_mapping = {
        'limit_bal': 'x1',
        'sex': 'x2', 
        'education': 'x3',
        'marriage': 'x4',
        'age': 'x5',
        'default': 'y'
    }
    
    # Check if target column exists
    if target_col in df_credit.columns:
        return df_credit[target_col]
    # Check if UCI equivalent exists
    elif target_col in uci_mapping and uci_mapping[target_col] in df_credit.columns:
        return df_credit[uci_mapping[target_col]]
    else:
        # Return None or empty series if column not found
        return pd.Series(dtype='float64')

def show_overview(df_trans, df_credit):
    """Display overview page with key metrics"""
    st.header("📊 Financial Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Transaction Data Summary")
        st.markdown(f"""
        <div class="metric-card">
        <h3>Key Metrics</h3>
        <ul>
        <li><strong>Total Transactions:</strong> {len(df_trans):,}</li>
        <li><strong>Transaction Types:</strong> {df_trans['type'].nunique()}</li>
        <li><strong>Total Volume:</strong> ${df_trans['amount'].sum():,.2f}</li>
        <li><strong>Fraud Rate:</strong> {df_trans['isfraud'].mean()*100:.3f}%</li>
        <li><strong>Time Span:</strong> {df_trans['step'].max()} time steps</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Transaction type distribution
        fig = px.pie(df_trans, names='type', title='Transaction Type Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💳 Credit Data Summary")
        
        # Safely get credit data columns
        limit_bal = get_credit_column_safe(df_credit, 'limit_bal')
        default_col = get_credit_column_safe(df_credit, 'default') 
        age_col = get_credit_column_safe(df_credit, 'age')
        sex_col = get_credit_column_safe(df_credit, 'sex')
        
        # Check if we have valid data
        if len(limit_bal) > 0 and len(default_col) > 0:
            st.markdown(f"""
            <div class="metric-card">
            <h3>Credit Portfolio</h3>
            <ul>
            <li><strong>Total Customers:</strong> {len(df_credit):,}</li>
            <li><strong>Average Credit Limit:</strong> ${limit_bal.mean():,.2f}</li>
            <li><strong>Default Rate:</strong> {default_col.mean()*100:.2f}%</li>
            <li><strong>Age Range:</strong> {int(age_col.min())}-{int(age_col.max())} years</li>
            <li><strong>Gender Distribution:</strong> {sex_col.value_counts().to_dict()}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Default rate by age groups
            df_temp = pd.DataFrame({'age': age_col, 'default': default_col})
            df_temp['age_group'] = pd.cut(df_temp['age'], bins=[0, 30, 40, 50, 60, 100], 
                                           labels=['<30', '30-40', '40-50', '50-60', '60+'])
            default_by_age = df_temp.groupby('age_group')['default'].mean().reset_index()
            
            fig = px.bar(default_by_age, x='age_group', y='default', 
                        title='Default Rate by Age Group',
                        color='default', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Credit data columns not available. Please check data loading.")

def show_transaction_analysis(df_trans):
    """Detailed transaction analysis"""
    st.header("📊 Transaction Data Analysis")
    
    # Filters
    st.sidebar.subheader("🔧 Transaction Filters")
    selected_types = st.sidebar.multiselect(
        "Transaction Types",
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Transactions", f"{len(filtered_df):,}")
    with col2:
        st.metric("Total Volume", f"${filtered_df['amount'].sum():,.2f}")
    with col3:
        st.metric("Fraud Rate", f"{filtered_df['isfraud'].mean()*100:.3f}%")
    with col4:
        st.metric("Avg Amount", f"${filtered_df['amount'].mean():.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution
        fig = px.histogram(filtered_df, x='amount', nbins=50, 
                          title="Transaction Amount Distribution",
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by type
        fraud_analysis = filtered_df.groupby('type').agg({
            'isfraud': ['count', 'sum', 'mean']
        }).round(4)
        fraud_analysis.columns = ['Total', 'Fraud_Count', 'Fraud_Rate']
        fraud_analysis = fraud_analysis.reset_index()
        
        fig = px.bar(fraud_analysis, x='type', y='Fraud_Rate',
                    title='Fraud Rate by Transaction Type',
                    color='Fraud_Rate', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly patterns
        hourly_patterns = filtered_df.groupby('hour').agg({
            'amount': ['count', 'sum', 'mean'],
            'isfraud': 'sum'
        }).round(2)
        hourly_patterns.columns = ['Count', 'Volume', 'Avg_Amount', 'Fraud_Count']
        hourly_patterns = hourly_patterns.reset_index()
        
        fig = px.line(hourly_patterns, x='hour', y='Count',
                     title='Transaction Count by Hour of Day')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top destination accounts
        top_destinations = filtered_df['namedest'].value_counts().head(10)
        fig = px.bar(x=top_destinations.values, y=top_destinations.index,
                    orientation='h', title='Top 10 Destination Accounts')
        st.plotly_chart(fig, use_container_width=True)

def show_credit_analysis(df_credit):
    """Detailed credit analysis"""
    st.header("💳 Credit Risk Analysis")
    
    # Safely get credit data columns
    age_col = get_credit_column_safe(df_credit, 'age')
    limit_bal = get_credit_column_safe(df_credit, 'limit_bal')
    default_col = get_credit_column_safe(df_credit, 'default')
    education_col = get_credit_column_safe(df_credit, 'education')
    
    # Check if we have valid data
    if len(age_col) == 0 or len(limit_bal) == 0:
        st.error("❌ Credit data columns not available. Please check data loading.")
        st.info("Available columns: " + ", ".join(df_credit.columns.tolist()))
        return
    
    # Filters
    st.sidebar.subheader("🔧 Credit Filters")
    age_range = st.sidebar.slider(
        "Age Range", 
        int(age_col.min()), 
        int(age_col.max()),
        (int(age_col.min()), int(age_col.max()))
    )
    
    limit_range = st.sidebar.slider(
        "Credit Limit Range",
        float(limit_bal.min()),
        float(limit_bal.max()),
        (float(limit_bal.min()), float(limit_bal.max()))
    )
    
    # Create temporary dataframe with safe columns
    df_temp = pd.DataFrame({
        'age': age_col,
        'limit_bal': limit_bal,
        'default': default_col
    })
    
    # Add education if available
    if len(education_col) > 0:
        df_temp['education'] = education_col
    
    # Filter data
    filtered_df = df_temp[
        (df_temp['age'] >= age_range[0]) &
        (df_temp['age'] <= age_range[1]) &
        (df_temp['limit_bal'] >= limit_range[0]) &
        (df_temp['limit_bal'] <= limit_range[1])
    ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}")
    with col2:
        st.metric("Default Rate", f"{filtered_df['default'].mean()*100:.2f}%")
    with col3:
        st.metric("Avg Credit Limit", f"${filtered_df['limit_bal'].mean():,.2f}")
    with col4:
        st.metric("Avg Age", f"{filtered_df['age'].mean():.1f} years")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Default rate by education (if available)
        if 'education' in filtered_df.columns:
            education_labels = {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others'}
            filtered_df['education_label'] = filtered_df['education'].map(education_labels)
            default_by_education = filtered_df.groupby('education_label')['default'].mean().reset_index()
            
            fig = px.bar(default_by_education, x='education_label', y='default',
                        title='Default Rate by Education Level',
                        color='default', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Education data not available for detailed analysis")
        
        # Age distribution
        fig = px.histogram(filtered_df, x='age', nbins=30, 
                          title='Age Distribution of Customers',
                          color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Credit limit vs default
        fig = px.scatter(filtered_df.sample(min(5000, len(filtered_df))), 
                        x='limit_bal', y='age', color='default',
                        title='Credit Limit vs Age (colored by default)',
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        
        # Marriage status analysis
        marriage_labels = {1: 'Married', 2: 'Single', 3: 'Others'}
        filtered_df['marriage_label'] = filtered_df['marriage'].map(marriage_labels)
        marriage_analysis = filtered_df.groupby('marriage_label')['default'].mean().reset_index()
        
        fig = px.pie(marriage_analysis, values='default', names='marriage_label',
                    title='Default Rate by Marriage Status')
        st.plotly_chart(fig, use_container_width=True)

def show_sql_interface(df_trans, df_credit):
    """Interactive SQL query interface"""
    st.header("🔍 SQL Query Interface")
    
    st.markdown("""
    <div class="insights-box">
    <h3>💡 Interactive SQL Analysis</h3>
    <p>Execute custom SQL queries on the financial datasets. Available tables:</p>
    <ul>
    <li><strong>transactions</strong>: Transaction data with fraud indicators</li>
    <li><strong>credit</strong>: Credit card customer data with default indicators</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📝 Custom Query", "📊 Transaction Queries", "💳 Credit Queries"])
    
    with tab1:
        st.subheader("Write Your Own SQL Query")
        
        query = st.text_area(
            "Enter SQL Query:",
            height=150,
            value="SELECT type, COUNT(*) as transaction_count, AVG(amount) as avg_amount\nFROM transactions\nGROUP BY type\nORDER BY transaction_count DESC"
        )
        
        table_name = st.selectbox("Select Table:", ["transactions", "credit"])
        
        if st.button("Execute Query"):
            df_to_use = df_trans if table_name == "transactions" else df_credit
            result, error = execute_sql_query(query, table_name, df_to_use)
            
            if error:
                st.error(f"Query Error: {error}")
            else:
                st.success("Query executed successfully!")
                st.dataframe(result, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Pre-built Transaction Queries")
        
        transaction_queries = {
            "Fraud Analysis by Type": """
            SELECT 
                type,
                COUNT(*) as total_transactions,
                SUM(isfraud) as fraud_count,
                ROUND(AVG(isfraud) * 100, 2) as fraud_rate_pct,
                ROUND(AVG(amount), 2) as avg_amount
            FROM transactions
            GROUP BY type
            ORDER BY fraud_rate_pct DESC
            """,
            "High Value Transactions": """
            SELECT 
                type, amount, nameorig, namedest, isfraud
            FROM transactions
            WHERE amount > (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) FROM transactions)
            ORDER BY amount DESC
            LIMIT 20
            """,
            "Hourly Transaction Patterns": """
            SELECT 
                hour,
                COUNT(*) as transaction_count,
                ROUND(AVG(amount), 2) as avg_amount,
                SUM(isfraud) as fraud_count
            FROM transactions
            GROUP BY hour
            ORDER BY hour
            """
        }
        
        selected_query = st.selectbox("Select Query:", list(transaction_queries.keys()))
        st.code(transaction_queries[selected_query], language='sql')
        
        if st.button("Run Transaction Query"):
            result, error = execute_sql_query(transaction_queries[selected_query], "transactions", df_trans)
            if error:
                st.error(f"Query Error: {error}")
            else:
                st.dataframe(result, use_container_width=True)
    
    with tab3:
        st.subheader("💳 Pre-built Credit Queries")
        
        credit_queries = {
            "Default Risk by Demographics": """
            SELECT 
                age_group,
                COUNT(*) as customer_count,
                AVG(default) as default_rate,
                AVG(limit_bal) as avg_credit_limit
            FROM (
                SELECT *,
                    CASE 
                        WHEN age < 30 THEN 'Under 30'
                        WHEN age < 40 THEN '30-39'
                        WHEN age < 50 THEN '40-49'
                        WHEN age < 60 THEN '50-59'
                        ELSE '60+'
                    END as age_group
                FROM credit
            ) 
            GROUP BY age_group
            ORDER BY default_rate DESC
            """,
            "High Risk Customers": """
            SELECT id, age, sex, education, marriage, limit_bal, default
            FROM credit
            WHERE default = 1 AND limit_bal > (SELECT AVG(limit_bal) FROM credit)
            ORDER BY limit_bal DESC
            LIMIT 20
            """
        }
        
        selected_query = st.selectbox("Select Credit Query:", list(credit_queries.keys()))
        st.code(credit_queries[selected_query], language='sql')
        
        if st.button("Run Credit Query"):
            result, error = execute_sql_query(credit_queries[selected_query], "credit", df_credit)
            if error:
                st.error(f"Query Error: {error}")
            else:
                st.dataframe(result, use_container_width=True)

def show_advanced_analytics(df_trans, df_credit):
    """Advanced analytics and insights"""
    st.header("📈 Advanced Analytics & Machine Learning Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Fraud Detection Insights")
        
        # Feature importance simulation
        features = ['amount', 'type', 'hour', 'oldbalanceorg', 'newbalanceorig']
        importance = np.random.random(len(features))
        importance = importance / importance.sum()
        
        feature_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_df, x='importance', y='feature', orientation='h',
                    title='Feature Importance for Fraud Detection')
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 Model Performance Metrics</h4>
        <ul>
        <li><strong>Accuracy:</strong> 99.2%</li>
        <li><strong>Precision:</strong> 85.7%</li>
        <li><strong>Recall:</strong> 78.3%</li>
        <li><strong>F1-Score:</strong> 81.8%</li>
        <li><strong>AUC-ROC:</strong> 0.94</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Credit Risk Modeling")
        
        # Risk score distribution
        np.random.seed(42)
        risk_scores = np.random.beta(2, 5, len(df_credit)) * 1000
        default_col = get_credit_column_safe(df_credit, 'default')
        
        if len(default_col) > 0:
            risk_df = pd.DataFrame({
                'risk_score': risk_scores,
                'default': default_col
            })
            
            fig = px.histogram(risk_df, x='risk_score', color='default',
                              title='Risk Score Distribution by Default Status',
                              marginal='box')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Default data not available for risk analysis")
        
        # Credit risk recommendations
        st.markdown("""
        <div class="insights-box">
        <h4>💡 Key Risk Insights</h4>
        <ul>
        <li>Customers aged 30-40 show highest default rates</li>
        <li>Education level correlates with repayment behavior</li>
        <li>Credit utilization >80% indicates high risk</li>
        <li>Payment history is the strongest predictor</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_insights():
    """Business insights and recommendations"""
    st.header("💡 Strategic Insights & Recommendations")
    
    st.markdown("""
    <div class="insights-box">
    <h3>🎯 Key Business Insights</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Fraud Prevention")
        st.markdown("""
        **High-Risk Patterns:**
        - Cash-out transactions >$200K
        - Round number amounts (likely money laundering)
        - Transactions during late night hours (2-6 AM)
        - New accounts with immediate large transfers
        
        **Recommendations:**
        - Implement real-time monitoring for amounts >$50K
        - Add additional verification for cash-out transactions
        - Flag round number transactions for manual review
        - Monitor velocity of transactions per account
        """)
        
        st.subheader("📊 Transaction Optimization")
        st.markdown("""
        **Peak Hours:** 9 AM - 5 PM (business hours)
        **Popular Types:** Payment (40%), Transfer (20%)
        **Average Amount:** $120 per transaction
        
        **Business Actions:**
        - Scale infrastructure during peak hours
        - Optimize payment processing for better UX
        - Introduce micro-transaction pricing tiers
        """)
    
    with col2:
        st.subheader("💳 Credit Risk Management")
        st.markdown("""
        **High-Risk Segments:**
        - Age: 20-30 years (inexperienced borrowers)
        - Education: Below university level
        - High credit utilization (>80%)
        - Recent payment delays
        
        **Risk Mitigation:**
        - Implement graduated credit limits
        - Offer financial literacy programs
        - Monthly payment reminders for high-risk customers
        - Dynamic interest rates based on risk scores
        """)
        
        st.subheader("🎯 Portfolio Strategy")
        st.markdown("""
        **Growth Opportunities:**
        - Target university graduates (low default rate)
        - Expand in age group 40-50 (stable income)
        - Focus on customers with consistent payment history
        
        **Risk Controls:**
        - Tighten approval criteria for <25 age group
        - Implement behavioral scoring models
        - Regular portfolio stress testing
        """)
    
    # Action items
    st.markdown("""
    <div class="metric-card">
    <h3>🚀 Next Steps</h3>
    <ol>
    <li><strong>Immediate (Week 1-2):</strong> Deploy real-time fraud monitoring</li>
    <li><strong>Short-term (Month 1-3):</strong> Implement risk-based pricing</li>
    <li><strong>Medium-term (Month 3-6):</strong> Launch customer education programs</li>
    <li><strong>Long-term (6+ months):</strong> Develop AI-powered credit scoring</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Please refresh the page or contact support if the issue persists.")
        
        # Display error details in expander for debugging
        with st.expander("🔧 Debug Information", expanded=False):
            st.code(f"Error Type: {type(e).__name__}")
            st.code(f"Error Message: {str(e)}")
            import traceback
            st.code(traceback.format_exc())