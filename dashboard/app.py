import streamlit as st
import pandas as pd
import os
import json
import httpx

st.set_page_config(page_title="LLM Gateway Portal", layout="wide", page_icon="⚡")

# Custom CSS for rich premium aesthetics (Curated sleek dark theme tokens)
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .status-closed { background-color: #064E3B; color: #34D399; }
    .status-open { background-color: #7F1D1D; color: #F87171; }
    .status-half-open { background-color: #78350F; color: #FBBF24; }
</style>
""", unsafe_allowed_html=True)

st.title("⚡ Enterprise LLM Gateway Portal")

tab1, tab2, tab3 = st.tabs(["💬 Chat Sandbox (SSE Stream)", "📊 System Metrics & Caching", "⚙️ Admin Control Center"])

admin_token = os.environ.get("ADMIN_API_KEY", "admin-default-secret")
headers = {"X-Admin-Token": admin_token}
gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8080")

with tab1:
    st.markdown("### Interactive Chat Sandbox")
    st.caption("This client sandbox utilizes unified SSE streaming from the LLM Gateway API.")
    
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("Client API Key", value="sk-test-tier-1", type="password")
    with col2:
        model_name = st.text_input("Target Model Heuristic", value="gpt-4o-mini")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Reset button for chat history
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Stream response inside assistant message
    if prompt := st.chat_input("Ask a reasoning, code, or standard query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                full_response = ""
                # Utilize dynamic streaming
                with httpx.stream(
                    "POST",
                    f"{gateway_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "messages": st.session_state.messages,
                        "stream": True
                    },
                    timeout=30.0
                ) as r:
                    if r.status_code != 200:
                        st.error(f"Error {r.status_code}: {r.read().decode('utf-8')}")
                    else:
                        for line in r.iter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(data_str)
                                    if "error" in chunk_data:
                                        st.error(chunk_data["error"]["message"])
                                        break
                                    content = chunk_data["choices"][0]["delta"].get("content", "")
                                    full_response += content
                                    message_placeholder.markdown(full_response + " ▌")
                                except Exception:
                                    continue
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab2:
    st.markdown("### System Metrics & Telemetry")
    
    try:
        budgets_resp = httpx.get(f"{gateway_url}/admin/budgets", headers=headers, timeout=5.0)
        requests_resp = httpx.get(f"{gateway_url}/admin/requests", headers=headers, timeout=5.0)
        cb_resp = httpx.get(f"{gateway_url}/admin/circuit-breakers", headers=headers, timeout=5.0)
        
        if budgets_resp.status_code != 200 or requests_resp.status_code != 200:
            st.error("Cannot load metrics. Verify your admin keys and ensure the gateway is active.")
        else:
            budgets_df = pd.DataFrame(budgets_resp.json())
            requests_df = pd.DataFrame(requests_resp.json())
            cb_df = pd.DataFrame(cb_resp.json())
            
            # Key statistics display
            st.subheader("Gateway Overview")
            total_reqs = len(requests_df) if not requests_df.empty else 0
            cache_hits = len(requests_df[requests_df["backend"] == "cache"]) if total_reqs > 0 and "backend" in requests_df.columns else 0
            hit_rate = (cache_hits / total_reqs * 100) if total_reqs > 0 else 0.0
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total API Requests", total_reqs)
            with col_m2:
                total_spend = requests_df["cost_usd"].sum() if total_reqs > 0 else 0.0
                st.metric("Total Spend (USD)", f"${total_spend:.4f}")
            with col_m3:
                st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
            with col_m4:
                avg_lat = requests_df["latency_ms"].mean() if total_reqs > 0 else 0.0
                st.metric("Average Latency", f"{avg_lat:.1f}ms")
                
            # Budget limits table
            st.subheader("Client Budgets & Rate Limits")
            if not budgets_df.empty:
                if "api_key" in budgets_df.columns:
                    budgets_df["masked_api_key"] = budgets_df["api_key"].apply(lambda k: f"{k[:5]}...{k[-4:]}" if len(k) > 10 else k)
                st.dataframe(budgets_df, use_container_width=True)
            else:
                st.info("No budgets configured.")

            # Spend charts
            if total_reqs > 0:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("#### Spend per Backend")
                    spend_grouped = requests_df.groupby("backend")["cost_usd"].sum().reset_index()
                    st.bar_chart(spend_grouped.set_index("backend"))
                with col_chart2:
                    st.markdown("#### Latency per Backend")
                    latency_grouped = requests_df.groupby("backend")["latency_ms"].mean().reset_index()
                    st.bar_chart(latency_grouped.set_index("backend"))
            
    except Exception as e:
        st.error(f"Failed to fetch telemetry: {e}")

with tab3:
    st.markdown("### Gateway Administrative Controls")
    
    try:
        # Dynamic active health status report
        st.subheader("🔍 Real-time Backend Health Checks")
        health_resp = httpx.get(f"{gateway_url}/health/backends", headers=headers, timeout=5.0)
        
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            h_cols = st.columns(max(1, len(health_data)))
            for idx, (backend_id, status) in enumerate(health_data.items()):
                with h_cols[idx % len(h_cols)]:
                    h_symbol = "🟢 Healthy" if status["healthy"] else "🔴 Unreachable"
                    cb_state = status["circuit_breaker_state"]
                    badge_style = "status-closed" if cb_state == "CLOSED" else "status-half-open" if cb_state == "HALF_OPEN" else "status-open"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5>{backend_id}</h5>
                        <p>Health: <strong>{h_symbol}</strong></p>
                        <p>Circuit State: <span class="status-badge {badge_style}">{cb_state}</span></p>
                    </div>
                    """, unsafe_allowed_html=True)
        else:
            st.warning("Could not fetch active health statuses.")
            
        # Admin Forms Section
        st.write("---")
        col_admin1, col_admin2 = st.columns(2)
        
        with col_admin1:
            st.subheader("Modify Client Budget / Rate Limits")
            with st.form("budget_update_form"):
                target_key = st.text_input("Target client API Key (or new key)")
                daily_limit = st.number_input("Daily Cost Limit (USD)", min_value=0.0, value=10.0, step=1.0)
                monthly_limit = st.number_input("Monthly Cost Limit (USD)", min_value=0.0, value=100.0, step=10.0)
                rpm_limit = st.number_input("Custom Limit (Requests/Minute)", min_value=1, value=60, step=5)
                
                submitted = st.form_submit_button("Update Client limits")
                if submitted:
                    if not target_key:
                        st.error("Please specify a target API key.")
                    else:
                        payload = {
                            "daily_limit_usd": daily_limit,
                            "monthly_limit_usd": monthly_limit,
                            "requests_per_minute": rpm_limit
                        }
                        update_resp = httpx.patch(
                            f"{gateway_url}/admin/budgets/{target_key}",
                            headers=headers,
                            json=payload,
                            timeout=5.0
                        )
                        if update_resp.status_code == 200:
                            st.success(f"Successfully configured credentials for key: {target_key}")
                            st.rerun()
                        else:
                            st.error(f"Failed to update: {update_resp.text}")
                            
        with col_admin2:
            st.subheader("Manual Circuit Breaker Overrides")
            cb_list_resp = httpx.get(f"{gateway_url}/admin/circuit-breakers", headers=headers, timeout=5.0)
            if cb_list_resp.status_code == 200:
                cb_list = cb_list_resp.json()
                if cb_list:
                    backends_list = [c["backend_id"] for c in cb_list]
                    selected_backend = st.selectbox("Select Backend to Reset", backends_list)
                    
                    if st.button("🔌 Reset Circuit Breaker (Force CLOSE)"):
                        reset_resp = httpx.post(
                            f"{gateway_url}/admin/circuit-breakers/{selected_backend}/reset",
                            headers=headers,
                            timeout=5.0
                        )
                        if reset_resp.status_code == 200:
                            st.success(f"Successfully closed circuit breaker for {selected_backend}")
                            st.rerun()
                        else:
                            st.error(f"Failed to reset: {reset_resp.text}")
                else:
                    st.info("No circuit breakers registered in state yet.")
            else:
                st.error("Could not fetch circuit breaker registry list.")
                
    except Exception as e:
        st.error(f"Admin diagnostics failure: {e}")
