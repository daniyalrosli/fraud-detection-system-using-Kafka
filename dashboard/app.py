"""
Real-Time Fraud Detection Dashboard.

Streamlit dashboard for visualizing fraud detection results,
transaction metrics, and alerts in real-time.
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = os.getenv(
    'DB_PATH', 
    str(Path(__file__).parent.parent / 'data' / 'predictions.db')
)
REFRESH_INTERVAL = int(os.getenv('REFRESH_INTERVAL', '3'))

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_database_connection():
    """Create a cached database connection."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_recent_data(minutes: int = 60) -> pd.DataFrame:
    """
    Load recent transaction data from database.
    
    Args:
        minutes: Number of minutes of history to load.
        
    Returns:
        DataFrame with recent transactions.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        query = f"""
            SELECT * FROM predictions 
            WHERE processed_at >= '{cutoff_time}'
            ORDER BY processed_at DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['processed_at'] = pd.to_datetime(df['processed_at'])
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def load_all_data() -> pd.DataFrame:
    """Load all transaction data from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM predictions ORDER BY processed_at DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['processed_at'] = pd.to_datetime(df['processed_at'])
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def render_metrics(df: pd.DataFrame) -> None:
    """Render key metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    fraud_count = df['is_fraud_predicted'].sum() if not df.empty else 0
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0
    avg_amount = df['amount'].mean() if not df.empty else 0
    high_risk = len(df[df['fraud_probability'] > 0.85]) if not df.empty else 0
    
    with col1:
        st.metric(
            label="📊 Total Transactions",
            value=f"{total:,}",
            delta=f"+{min(total, 10)} recent"
        )
    
    with col2:
        st.metric(
            label="🚨 Fraud Detected",
            value=f"{fraud_count:,}",
            delta=f"{fraud_rate:.1f}%"
        )
    
    with col3:
        st.metric(
            label="💰 Avg Transaction",
            value=f"${avg_amount:,.2f}"
        )
    
    with col4:
        st.metric(
            label="⚠️ High Risk Alerts",
            value=f"{high_risk:,}",
            delta="Prob > 85%"
        )


def render_fraud_gauge(df: pd.DataFrame) -> None:
    """Render fraud rate gauge chart."""
    if df.empty:
        st.info("No data available for gauge")
        return
    
    # Calculate fraud rate for last 5 minutes
    cutoff = datetime.now() - timedelta(minutes=5)
    recent = df[df['processed_at'] >= cutoff]
    
    if len(recent) > 0:
        fraud_rate = recent['is_fraud_predicted'].mean() * 100
    else:
        fraud_rate = 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fraud_rate,
        title={'text': "Fraud Rate (Last 5 min)", 'font': {'size': 20}},
        delta={'reference': 5, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkred"},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 5], 'color': 'lightgreen'},
                {'range': [5, 15], 'color': 'yellow'},
                {'range': [15, 100], 'color': 'salmon'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 10
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_transaction_volume(df: pd.DataFrame) -> None:
    """Render transaction volume over time."""
    if df.empty:
        st.info("No data available for volume chart")
        return
    
    # Resample by minute
    df_copy = df.set_index('processed_at').copy()
    volume = df_copy.resample('1min').size().reset_index(name='count')
    volume.columns = ['Time', 'Transactions']
    
    fig = px.line(
        volume,
        x='Time',
        y='Transactions',
        title='Transaction Volume Over Time',
        markers=True
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Transactions",
        height=350
    )
    fig.update_traces(line_color='#1f77b4', line_width=2)
    st.plotly_chart(fig, use_container_width=True)


def render_fraud_comparison(df: pd.DataFrame) -> None:
    """Render fraud vs legitimate transaction comparison."""
    if df.empty:
        st.info("No data available for comparison chart")
        return
    
    counts = df['is_fraud_predicted'].value_counts().reset_index()
    counts.columns = ['Type', 'Count']
    counts['Type'] = counts['Type'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.bar(
        counts,
        x='Type',
        y='Count',
        title='Fraud vs Legitimate Transactions',
        color='Type',
        color_discrete_map={'Legitimate': '#2ecc71', 'Fraudulent': '#e74c3c'}
    )
    fig.update_layout(
        showlegend=False,
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


def render_merchant_fraud_rate(df: pd.DataFrame) -> None:
    """Render fraud rate by merchant category."""
    if df.empty:
        st.info("No data available for merchant analysis")
        return
    
    merchant_stats = df.groupby('merchant_category').agg({
        'is_fraud_predicted': ['sum', 'count']
    }).reset_index()
    merchant_stats.columns = ['Category', 'Fraud', 'Total']
    merchant_stats['Fraud Rate (%)'] = (
        merchant_stats['Fraud'] / merchant_stats['Total'] * 100
    ).round(1)
    merchant_stats = merchant_stats.sort_values('Fraud Rate (%)', ascending=True)
    
    fig = px.bar(
        merchant_stats,
        x='Fraud Rate (%)',
        y='Category',
        orientation='h',
        title='Fraud Rate by Merchant Category',
        color='Fraud Rate (%)',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_live_feed(df: pd.DataFrame, n_rows: int = 20) -> None:
    """Render live transaction feed table."""
    if df.empty:
        st.info("No transactions to display")
        return
    
    display_df = df.head(n_rows)[[
        'transaction_id', 'timestamp', 'amount', 'merchant_category',
        'is_fraud_predicted', 'fraud_probability'
    ]].copy()
    
    display_df.columns = [
        'Transaction ID', 'Time', 'Amount', 'Merchant',
        'Fraud', 'Probability'
    ]
    display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.2f}")
    display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.1%}")
    display_df['Fraud'] = display_df['Fraud'].apply(lambda x: '🚨 Yes' if x else '✓ No')
    display_df['Time'] = pd.to_datetime(display_df['Time']).dt.strftime('%H:%M:%S')
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_alerts(df: pd.DataFrame) -> None:
    """Render high-risk transaction alerts."""
    if df.empty:
        return
    
    high_risk = df[df['fraud_probability'] > 0.85].head(5)
    
    if not high_risk.empty:
        st.error("🚨 **HIGH RISK ALERTS** - Transactions with fraud probability > 85%")
        
        for _, row in high_risk.iterrows():
            st.warning(
                f"**Transaction {row['transaction_id']}** | "
                f"Amount: ${row['amount']:,.2f} | "
                f"Merchant: {row['merchant_category']} | "
                f"Probability: {row['fraud_probability']:.1%}"
            )


def main():
    """Main dashboard application."""
    # Header
    st.title("🔒 Real-Time Fraud Detection Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        time_range = st.selectbox(
            "Time Range",
            options=[5, 15, 30, 60, 120],
            index=2,
            format_func=lambda x: f"Last {x} minutes"
        )
        
        st.markdown("---")
        st.header("📈 Statistics")
        
        # Load all data for stats
        all_data = load_all_data()
        if not all_data.empty:
            st.write(f"**Total Records:** {len(all_data):,}")
            st.write(f"**Fraud Rate:** {all_data['is_fraud_predicted'].mean():.1%}")
            st.write(f"**Avg Probability:** {all_data['fraud_probability'].mean():.1%}")
        
        st.markdown("---")
        st.info(f"🔄 Auto-refresh: {REFRESH_INTERVAL}s")
        
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Load recent data
    df = load_recent_data(minutes=time_range)
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        st.warning(
            "⚠️ Database not found. Please ensure the Kafka consumer "
            "is running and processing transactions."
        )
        st.info(f"Expected path: `{DB_PATH}`")
        st.stop()
    
    if df.empty:
        st.info(
            "📊 No recent transactions found. The dashboard will update "
            "automatically when data is available."
        )
    
    # Render alerts at the top
    render_alerts(df)
    
    # Key metrics
    render_metrics(df)
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        render_fraud_gauge(df)
    with col2:
        render_fraud_comparison(df)
    
    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        render_transaction_volume(df)
    with col4:
        render_merchant_fraud_rate(df)
    
    st.markdown("---")
    
    # Live feed
    st.subheader("📋 Live Transaction Feed")
    render_live_feed(df)
    
    # Auto-refresh
    import time
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
