import streamlit as st
import sqlite3
import pandas as pd
import os
import httpx

DB_PATH = os.environ.get("DB_PATH", "ledger.db")

st.set_page_config(page_title="LLM Gateway UI", layout="wide")

st.title("LLM Gateway & Dashboard")

tab1, tab2 = st.tabs(["💬 Chat Sandbox", "📊 Cost Dashboard"])

with tab1:
    st.markdown("### Test the Gateway API")
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("API Key", value="sk-test-tier-1")
    with col2:
        model_name = st.text_input("Model Request", value="gpt-4o-mini")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Say something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = httpx.post(
                    "http://localhost:8080/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "messages": st.session_state.messages
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    ai_reply = data["choices"][0]["message"]["content"]
                    backend_used = data.get("backend", "unknown")
                    message_placeholder.markdown(f"{ai_reply}\n\n*(Routed to backend: {backend_used})*")
                    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                else:
                    message_placeholder.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                message_placeholder.error(f"Connection failed. Is the Gateway running on port 8080? Error: {e}")

with tab2:
    def get_db_connection():
        path = DB_PATH
        if not os.path.exists(path):
            return None
        return sqlite3.connect(path)

    conn = get_db_connection()
    if conn is None:
        st.warning("No ledger database found yet. Send a message in the Chat Sandbox first!")
    else:
        try:
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
