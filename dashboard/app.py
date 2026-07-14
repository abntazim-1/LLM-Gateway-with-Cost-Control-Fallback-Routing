import streamlit as st
import sqlite3
import pandas as pd
import os

DB_PATH = os.environ.get("DB_PATH", "ledger.db")

st.set_page_config(page_title="LLM Gateway Dashboard", layout="wide")

st.title("LLM Gateway Enterprise Dashboard")

def get_db_connection():
    # Streamlit runs from the root usually, check if ledger.db exists
    path = DB_PATH
    if not os.path.exists(path):
        st.error(f"Ledger database not found at {path}. Make sure the gateway has processed requests.")
        st.stop()
    return sqlite3.connect(path)

try:
    conn = get_db_connection()
    
    st.header("Budget Overview")
    budgets_df = pd.read_sql_query("SELECT api_key, daily_limit_usd, spend_today, monthly_limit_usd, spend_month FROM budgets", conn)
    st.dataframe(budgets_df, use_container_width=True)

    st.header("Recent Requests (Cost & Routing)")
    requests_df = pd.read_sql_query("SELECT id, api_key, backend, model, cost_usd, latency_ms, timestamp FROM requests ORDER BY timestamp DESC LIMIT 50", conn)
    st.dataframe(requests_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spend by Backend")
        if not requests_df.empty:
            spend_by_backend = requests_df.groupby("backend")["cost_usd"].sum().reset_index()
            st.bar_chart(spend_by_backend.set_index("backend"))
        else:
            st.write("No data yet.")
            
    with col2:
        st.subheader("Latency by Backend (ms)")
        if not requests_df.empty:
            latency_by_backend = requests_df.groupby("backend")["latency_ms"].mean().reset_index()
            st.bar_chart(latency_by_backend.set_index("backend"))
        else:
            st.write("No data yet.")

    conn.close()

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
