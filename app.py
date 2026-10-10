"""
app.py - Interactive Streamlit Dashboard for WTQ 2026 Utility Bill Decoder.
Provides visual verification, bill extraction inspection, grounded Q&A, and CSV export.
"""

import os
import json
import streamlit as st
from PIL import Image

from model_client import GeminiModelClient, ALLOWED_MODELS
from extractor import BillExtractor
from question_engine import QuestionEngine
from schemas import Level1Output, InternalEvidence


st.set_page_config(
    page_title="WTQ 2026 - Utility Bill Decoder",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for polished competition styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ WTQ 2026 — Utility Bill Decoder</div>', unsafe_allow_html=True)
st.caption("AI-powered, grounded utility bill extractor and customer Q&A engine for KE, LESCO, and IESCO bills.")

# Sidebar configuration
with st.sidebar:
    st.subheader("Model Configuration")
    selected_model = st.selectbox("Approved Gemini Model", ALLOWED_MODELS, index=0)
    use_cache = st.checkbox("Enable LLM Response Cache", value=True)
    st.divider()
    st.info("Ensure `GOOGLE_API_KEY_PRIMARY` is configured in `.env`.")

# Initialize client & pipeline
@st.cache_resource
def get_pipeline(model_name: str, cache_enabled: bool):
    client = GeminiModelClient(model_name=model_name, use_cache=cache_enabled)
    extractor = BillExtractor(client=client)
    q_engine = QuestionEngine(client=client)
    return extractor, q_engine

try:
    extractor, q_engine = get_pipeline(selected_model, use_cache)
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Electricity Bill (PNG / JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    temp_dir = os.path.join(os.path.dirname(__file__), ".temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Uploaded Bill Image")
        st.image(temp_path, use_container_width=True)

    with col2:
        st.subheader("Extraction & Analysis")
        if st.button("Extract Bill Data", type="primary") or "extracted_data" in st.session_state:
            if "extracted_data" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
                with st.spinner("Analyzing bill with Gemini vision model..."):
                    try:
                        l1, evidence, diags = extractor.extract_bill(temp_path)
                        st.session_state["extracted_l1"] = l1
                        st.session_state["extracted_evidence"] = evidence
                        st.session_state["extracted_diags"] = diags
                        st.session_state["last_uploaded"] = uploaded_file.name
                        st.session_state["extracted_data"] = True
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                        st.stop()

            l1: Level1Output = st.session_state["extracted_l1"]
            evidence: InternalEvidence = st.session_state["extracted_evidence"]

            # Key metrics summary
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Provider", l1.provider or "N/A")
            m2.metric("Bill Month", l1.bill_month or "N/A")
            m3.metric("Units Consumed", f"{l1.units_consumed or 0} kWh")
            m4.metric("Current Bill", f"Rs. {l1.current_bill or 0:,.2f}" if l1.current_bill else "N/A")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Due Date", l1.due_date or "N/A")
            d2.metric("Total Charges", f"Rs. {l1.total_charges or 0:,.2f}" if l1.total_charges else "N/A")
            d3.metric("Total Taxes", f"Rs. {l1.total_taxes or 0:,.2f}" if l1.total_taxes else "N/A")
            if l1.payable_within_due_date is not None:
                if l1.payable_within_due_date < 0:
                    d4.metric("Payable Status", f"-Rs. {abs(l1.payable_within_due_date):,.0f} (Credit / Do Not Pay)")
                elif l1.payable_within_due_date == 0:
                    d4.metric("Payable", "Rs. 0 (Zero Balance)")
                else:
                    d4.metric("Payable", f"Rs. {l1.payable_within_due_date:,.0f}")
            else:
                d4.metric("Payable", "N/A")

            tab1, tab2, tab3 = st.tabs(["Level 1 JSON", "Line Items", "Ask a Question"])

            with tab1:
                l1_dict = l1.to_level1_dict()
                st.json(l1_dict)
                st.download_button(
                    "Download Level 1 JSON",
                    data=json.dumps(l1_dict, indent=2),
                    file_name=f"{l1.provider or 'bill'}_level1.json",
                    mime="application/json"
                )

            with tab2:
                col_c, col_t = st.columns(2)
                with col_c:
                    st.write("**Charges Breakdown:**")
                    if l1.charges:
                        for c in l1.charges:
                            st.write(f"- `{c.type}`: Rs. {c.amount}")
                    else:
                        st.caption("No itemized charges found.")
                with col_t:
                    st.write("**Taxes Breakdown:**")
                    if l1.taxes:
                        for t in l1.taxes:
                            st.write(f"- `{t.type}`: Rs. {t.amount}")
                    else:
                        st.caption("No itemized taxes found.")

            with tab3:
                st.write("**Grounded Customer Q&A**")
                user_question = st.text_input("Ask any question regarding this bill:", placeholder="e.g. What percentage of the bill is taxes?")
                if st.button("Submit Question") and user_question:
                    with st.spinner("Generating bill-grounded answer..."):
                        ans = q_engine.answer_question(user_question, l1, evidence)
                        st.success(ans)
