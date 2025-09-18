"""
Financial Data Analytics - Cloud Deployment Version
Simplified single-file Streamlit app for easy cloud deployment
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

@st.cache_data
def generate_sample_data():
    """Generate sample financial data for demonstration"""
    np.random.seed(42)
    
    # Generate transaction data
    n_transactions = 10000
    transactions = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'amount': np.random.lognormal(3, 1.5, n_transactions),
        'transaction_type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_transactions),
        'account_balance_before': np.random.uniform(0, 50000, n_transactions),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.998, 0.002]),
        'customer_id': np.random.randint(1, 1000, n_transactions),
        'merchant_id': np.random.randint(1, 500, n_transactions),
        'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='H')
    })
    
    # Generate credit data
    n_customers = 1000
    credit_data = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'credit_limit': np.random.uniform(1000, 50000, n_customers),
        'balance': np.random.uniform(0, 30000, n_customers),
        'payment_history': np.random.choice([0, 1, 2], n_customers, p=[0.7, 0.25, 0.05]),
        'default_payment': np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),
        'age': np.random.randint(18, 80, n_customers),
        'education_level': np.random.choice([1, 2, 3, 4], n_customers),
        'marriage_status': np.random.choice([1, 2, 3], n_customers)
    })
    
    return transactions, credit_data

def main():
    st.markdown("""
    <div style='text-align: center'>
        <h1 style='color: #1f77b4'>💰 Financial Data Analytics Dashboard</h1>
        <p style='font-size: 1.2em; color: #666'>Interactive analysis of financial transactions and credit risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    transactions, credit_data = generate_sample_data()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "📊 Transaction Analysis", 
        "💳 Credit Risk Analysis", 
        "🔍 SQL Query Interface",
        "📈 Advanced Analytics"
    ])
    
    if page == "📊 Transaction Analysis":
        st.header("Transaction Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(transactions):,}")
        with col2:
            st.metric("Total Volume", f"${transactions['amount'].sum():,.2f}")
        with col3:
            st.metric("Avg Transaction", f"${transactions['amount'].mean():.2f}")
        with col4:
            fraud_rate = transactions['is_fraud'].mean() * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        # Transaction volume over time
        st.subheader("Transaction Volume Over Time")
        daily_volume = transactions.groupby(transactions['timestamp'].dt.date)['amount'].sum().reset_index()
        fig = px.line(daily_volume, x='timestamp', y='amount', title='Daily Transaction Volume')
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction type distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transaction Types")
            type_dist = transactions['transaction_type'].value_counts()
            fig = px.pie(values=type_dist.values, names=type_dist.index, title='Transaction Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Amount Distribution")
            fig = px.histogram(transactions, x='amount', nbins=50, title='Transaction Amount Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "💳 Credit Risk Analysis":
        st.header("Credit Risk Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(credit_data):,}")
        with col2:
            avg_balance = credit_data['balance'].mean()
            st.metric("Avg Balance", f"${avg_balance:,.2f}")
        with col3:
            default_rate = credit_data['default_payment'].mean() * 100
            st.metric("Default Rate", f"{default_rate:.1f}%")
        with col4:
            avg_limit = credit_data['credit_limit'].mean()
            st.metric("Avg Credit Limit", f"${avg_limit:,.2f}")
        
        # Risk analysis
        st.subheader("Credit Risk Distribution")
        credit_data['utilization_rate'] = credit_data['balance'] / credit_data['credit_limit']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(credit_data, x='utilization_rate', nbins=30, 
                             title='Credit Utilization Rate Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(credit_data, x='credit_limit', y='balance', 
                           color='default_payment', title='Credit Limit vs Balance')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 SQL Query Interface":
        st.header("SQL Query Interface")
        st.write("Execute SQL queries on the financial datasets")
        
        # Create DuckDB connection and load data
        conn = duckdb.connect(':memory:')
        conn.execute("CREATE TABLE transactions AS SELECT * FROM transactions")
        conn.execute("CREATE TABLE credit_data AS SELECT * FROM credit_data")
        
        # Sample queries
        st.subheader("Sample Queries")
        sample_queries = {
            "High Value Transactions": """
            SELECT transaction_type, COUNT(*) as count, AVG(amount) as avg_amount
            FROM transactions 
            WHERE amount > 1000
            GROUP BY transaction_type
            ORDER BY avg_amount DESC;
            """,
            "Fraud Analysis": """
            SELECT transaction_type, 
                   COUNT(*) as total_transactions,
                   SUM(is_fraud) as fraud_count,
                   ROUND(SUM(is_fraud) * 100.0 / COUNT(*), 2) as fraud_rate_pct
            FROM transactions
            GROUP BY transaction_type
            ORDER BY fraud_rate_pct DESC;
            """,
            "Credit Risk Segments": """
            SELECT 
                CASE 
                    WHEN balance/credit_limit < 0.3 THEN 'Low Risk'
                    WHEN balance/credit_limit < 0.7 THEN 'Medium Risk'
                    ELSE 'High Risk'
                END as risk_segment,
                COUNT(*) as customers,
                AVG(default_payment) as default_rate
            FROM credit_data
            GROUP BY risk_segment;
            """
        }
        
        selected_query = st.selectbox("Choose a sample query:", list(sample_queries.keys()))
        
        # Query input
        query = st.text_area("SQL Query:", value=sample_queries[selected_query], height=150)
        
        if st.button("Execute Query"):
            try:
                # Make data available to DuckDB
                conn.register('transactions', transactions)
                conn.register('credit_data', credit_data)
                
                result = conn.execute(query).fetchdf()
                st.success("Query executed successfully!")
                st.dataframe(result)
                
                # Download button
                csv = result.to_csv(index=False)
                st.download_button("Download Results", csv, "query_results.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Query error: {str(e)}")
    
    elif page == "📈 Advanced Analytics":
        st.header("Advanced Analytics & Insights")
        
        # Fraud prediction model simulation
        st.subheader("Fraud Detection Model Performance")
        
        # Simulate model metrics
        metrics_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
            'Current Model': [0.85, 0.78, 0.81, 0.94],
            'Industry Benchmark': [0.75, 0.70, 0.72, 0.88]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        fig = px.bar(metrics_df, x='Metric', y=['Current Model', 'Industry Benchmark'],
                     title='Model Performance vs Industry Benchmark', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk scoring
        st.subheader("Customer Risk Scoring")
        
        # Calculate risk scores
        credit_data['risk_score'] = (
            credit_data['balance'] / credit_data['credit_limit'] * 0.4 +
            credit_data['payment_history'] * 0.3 +
            (credit_data['age'] < 25).astype(int) * 0.3
        )
        
        fig = px.histogram(credit_data, x='risk_score', nbins=30, 
                          title='Customer Risk Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Business impact
        st.subheader("💡 Business Impact Insights")
        st.markdown("""
        <div style='background-color: #e6f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #0066cc'>
        <h4>🎯 Key Insights & Recommendations</h4>
        <ul>
        <li><strong>Fraud Detection:</strong> Current model shows 85% precision vs 75% industry average</li>
        <li><strong>Risk Management:</strong> 15% of customers are in high-risk category</li>
        <li><strong>Revenue Opportunity:</strong> Optimizing credit limits could increase revenue by 12%</li>
        <li><strong>Cost Savings:</strong> Enhanced fraud detection saves approximately $2.3M annually</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()