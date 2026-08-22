import json
import os

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env from the project root (works regardless of CWD)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"), override=False)


st.set_page_config(page_title="LLM Gateway Portal", layout="wide", page_icon="⚡")

# Custom CSS for rich premium aesthetics (Curated sleek dark theme tokens)
st.markdown(
    """
<style>
    .stApp {
        background: #0B0F19;
        color: #F1F5F9;
    }
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1F2937;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 12px;
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
""",
    unsafe_allow_html=True,
)

st.title("⚡ Enterprise LLM Gateway Portal")

with st.sidebar:
    st.header("🔑 Connection Settings")
    gateway_url = st.text_input(
        "Gateway URL", value=os.environ.get("GATEWAY_URL", "http://localhost:8080")
    )
    admin_token = st.text_input(
        "Admin Secret Key",
        value=os.environ.get("ADMIN_API_KEY", ""),
        type="password",
        help="Must match ADMIN_API_KEY configured on the Gateway.",
    )

headers = {"X-Admin-Token": admin_token} if admin_token else {}

tab1, tab2, tab3 = st.tabs(
    [
        "💬 Chat Sandbox (SSE Stream)",
        "📊 System Metrics & Caching",
        "⚙️ Admin Control Center",
    ]
)

with tab1:
    st.markdown("### Interactive Chat Sandbox")
    st.caption(
        "This client sandbox utilizes unified SSE streaming from the LLM Gateway API."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        api_key = st.text_input(
            "Client API Key",
            value=os.environ.get("CLIENT_API_KEY_TIER1", "sk-test-tier-1"),
            type="password",
        )
    with col2:
        model_name = st.text_input(
            "Target Model Heuristic",
            value=os.environ.get("DEFAULT_MODEL", "qwen2.5:0.5b"),
        )
    with col3:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=4096,
            value=256,
            step=32,
            help="Caps generation length so local/slow backends can't run "
            "past the client's own request timeout below.",
        )

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

    # If the last turn is an unanswered user message, generate the reply now
    # — this runs (and renders) before st.chat_input below, so the finished
    # exchange always ends up above the input box, not after it. st.chat_input
    # is nested inside this tab rather than called at the top level, so
    # Streamlit can't auto-pin it to the page bottom; this ordering is what
    # keeps it looking pinned instead of stranding it above the latest reply.
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                full_response = ""
                with httpx.stream(
                    "POST",
                    f"{gateway_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "messages": st.session_state.messages,
                        "stream": True,
                        "max_tokens": max_tokens,
                    },
                    timeout=60.0,
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
                                    content = chunk_data["choices"][0]["delta"].get(
                                        "content", ""
                                    )
                                    full_response += content
                                    message_placeholder.markdown(full_response + " ▌")
                                except Exception:
                                    continue
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": full_response}
                        )
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Capturing a new prompt and rerunning immediately — with nothing else in
    # between — is the safe pattern here. Doing any network/streaming work
    # before the rerun risks st.chat_input (nested in this tab, so its
    # per-submission value isn't cleared by Streamlit's normal top-level
    # auto-pin handling) replaying the same prompt on the next run.
    if prompt := st.chat_input("Ask a reasoning, code, or standard query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

with tab2:
    st.markdown("### System Metrics & Telemetry")

    try:
        budgets_resp = httpx.get(
            f"{gateway_url}/admin/budgets", headers=headers, timeout=5.0
        )
        requests_resp = httpx.get(
            f"{gateway_url}/admin/requests", headers=headers, timeout=5.0
        )
        cb_resp = httpx.get(
            f"{gateway_url}/admin/circuit-breakers", headers=headers, timeout=5.0
        )

        if budgets_resp.status_code != 200 or requests_resp.status_code != 200:
            st.error(
                "Cannot load metrics. Verify your admin keys and ensure the gateway is active."
            )
        else:
            budgets_df = pd.DataFrame(budgets_resp.json())
            requests_df = pd.DataFrame(requests_resp.json())
            cb_df = pd.DataFrame(cb_resp.json())

            # Key statistics display
            st.subheader("Gateway Overview")
            total_reqs = len(requests_df) if not requests_df.empty else 0
            cache_hits = (
                len(requests_df[requests_df["backend"] == "cache"])
                if total_reqs > 0 and "backend" in requests_df.columns
                else 0
            )
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
                    budgets_df["masked_api_key"] = budgets_df["api_key"].apply(
                        lambda k: f"{k[:5]}...{k[-4:]}" if len(k) > 10 else k
                    )
                st.dataframe(budgets_df, use_container_width=True)
            else:
                st.info("No budgets configured.")

            # Raw ledger view — every request logged by the gateway, most
            # recent first. The table widget's built-in toolbar (top-right on
            # hover) includes a CSV download button for free.
            st.subheader("Recent Requests (Ledger)")
            if not requests_df.empty:
                ledger_view = requests_df.copy()
                if "api_key" in ledger_view.columns:
                    ledger_view["api_key"] = ledger_view["api_key"].apply(
                        lambda k: f"{k[:5]}...{k[-4:]}" if len(k) > 10 else k
                    )
                if "timestamp" in ledger_view.columns:
                    ledger_view = ledger_view.sort_values("timestamp", ascending=False)
                st.dataframe(ledger_view, use_container_width=True, hide_index=True)
            else:
                st.info("No requests recorded yet.")

            # Spend charts
            if total_reqs > 0:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("#### Spend per Backend")
                    spend_grouped = (
                        requests_df.groupby("backend")["cost_usd"].sum().reset_index()
                    )
                    st.bar_chart(spend_grouped.set_index("backend"))
                with col_chart2:
                    st.markdown("#### Latency per Backend")
                    latency_grouped = (
                        requests_df.groupby("backend")["latency_ms"]
                        .mean()
                        .reset_index()
                    )
                    st.bar_chart(latency_grouped.set_index("backend"))

    except Exception as e:
        st.error(f"Failed to fetch telemetry: {e}")

with tab3:
    st.markdown("### Gateway Administrative Controls")

    try:
        # Dynamic active health status report
        st.subheader("🔍 Real-time Backend Health Checks")
        health_resp = httpx.get(
            f"{gateway_url}/health/backends", headers=headers, timeout=5.0
        )

        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.caption(
                "💰 marks the backend the `cost_first` routing strategy will "
                "pick, all else equal — the lowest `cost_per_1k_prompt` "
                "among healthy backends."
            )
            cheapest_id = None
            priced = {
                bid: s["cost_per_1k_prompt"]
                for bid, s in health_data.items()
                if "cost_per_1k_prompt" in s
            }
            if priced:
                cheapest_id = min(priced, key=priced.get)

            h_cols = st.columns(max(1, len(health_data)))
            for idx, (backend_id, status) in enumerate(health_data.items()):
                with h_cols[idx % len(h_cols)]:
                    h_symbol = "🟢 Healthy" if status["healthy"] else "🔴 Unreachable"
                    cb_state = status["circuit_breaker_state"]
                    badge_style = (
                        "status-closed"
                        if cb_state == "CLOSED"
                        else (
                            "status-half-open"
                            if cb_state == "HALF_OPEN"
                            else "status-open"
                        )
                    )
                    cheapest_badge = " 💰 Cheapest" if backend_id == cheapest_id else ""
                    prompt_cost = status.get("cost_per_1k_prompt")
                    completion_cost = status.get("cost_per_1k_completion")
                    cost_line = (
                        f"<p>Cost: <strong>${prompt_cost:.4f}</strong> / 1k prompt "
                        f"tokens, <strong>${completion_cost:.4f}</strong> / 1k "
                        f"completion tokens{cheapest_badge}</p>"
                        if prompt_cost is not None
                        else ""
                    )

                    st.markdown(
                        f"""
                    <div class="metric-card">
                        <h5>{backend_id}</h5>
                        <p>Health: <strong>{h_symbol}</strong></p>
                        <p>Circuit State: <span class="status-badge {badge_style}">{cb_state}</span></p>
                        {cost_line}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("Could not fetch active health statuses.")

        # Admin Forms Section
        st.write("---")
        col_admin1, col_admin2 = st.columns(2)

        with col_admin1:
            st.subheader("Modify Client Budget / Rate Limits")
            with st.form("budget_update_form"):
                target_key = st.text_input("Target client API Key (or new key)")
                daily_limit = st.number_input(
                    "Daily Cost Limit (USD)", min_value=0.0, value=10.0, step=1.0
                )
                monthly_limit = st.number_input(
                    "Monthly Cost Limit (USD)", min_value=0.0, value=100.0, step=10.0
                )
                rpm_limit = st.number_input(
                    "Custom Limit (Requests/Minute)", min_value=1, value=60, step=5
                )

                submitted = st.form_submit_button("Update Client limits")
                if submitted:
                    if not target_key:
                        st.error("Please specify a target API key.")
                    else:
                        payload = {
                            "daily_limit_usd": daily_limit,
                            "monthly_limit_usd": monthly_limit,
                            "requests_per_minute": rpm_limit,
                        }
                        update_resp = httpx.patch(
                            f"{gateway_url}/admin/budgets/{target_key}",
                            headers=headers,
                            json=payload,
                            timeout=5.0,
                        )
                        if update_resp.status_code == 200:
                            st.success(
                                f"Successfully configured credentials for key: {target_key}"
                            )
                            st.rerun()
                        else:
                            st.error(f"Failed to update: {update_resp.text}")

        with col_admin2:
            st.subheader("Manual Circuit Breaker Overrides")
            cb_list_resp = httpx.get(
                f"{gateway_url}/admin/circuit-breakers", headers=headers, timeout=5.0
            )
            if cb_list_resp.status_code == 200:
                cb_list = cb_list_resp.json()
                if cb_list:
                    backends_list = [c["backend_id"] for c in cb_list]
                    selected_backend = st.selectbox(
                        "Select Backend to Reset", backends_list
                    )

                    if st.button("🔌 Reset Circuit Breaker (Force CLOSE)"):
                        reset_resp = httpx.post(
                            f"{gateway_url}/admin/circuit-breakers/{selected_backend}/reset",
                            headers=headers,
                            timeout=5.0,
                        )
                        if reset_resp.status_code == 200:
                            st.success(
                                f"Successfully closed circuit breaker for {selected_backend}"
                            )
                            st.rerun()
                        else:
                            st.error(f"Failed to reset: {reset_resp.text}")
                else:
                    st.info("No circuit breakers registered in state yet.")
            else:
                st.error("Could not fetch circuit breaker registry list.")

    except Exception as e:
        st.error(f"Admin diagnostics failure: {e}")
