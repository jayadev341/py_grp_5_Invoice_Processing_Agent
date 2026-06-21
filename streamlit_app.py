import os
import streamlit as st
import uuid
import pandas as pd
from app import app

# Ensure temporary directory exists for uploads
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Enterprise Invoice Audit Agent", layout="wide")

st.title("📋 Automated Invoice Auditing & 3-Way Match Dashboard")
st.write("Ingest invoices, extract data via Docling + GPT-4o, run RAG compliance checks, and manage human-in-the-loop overrides.")

# Initialize session state for thread tracking and execution state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph_paused" not in st.session_state:
    st.session_state.graph_paused = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# -----------------------------------------------------------------
# SIDEBAR: Document Ingestion Controls
# -----------------------------------------------------------------
with st.sidebar:
    st.header("1. Ingestion Control Room")
    uploaded_file = st.file_uploader("Upload Vendor Invoice", type=["pdf", "png", "jpg"])
    
    if uploaded_file and uploaded_file.name != st.session_state.current_file:
        st.session_state.current_file = uploaded_file.name
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph_paused = False
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.file_path = file_path
        st.session_state.file_format = uploaded_file.name.split(".")[-1]
        st.success(f"Saved: {uploaded_file.name}")

    st.markdown("---")
    if st.button("Reset Audit Session Tracker", type="secondary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph_paused = False
        st.rerun()

# -----------------------------------------------------------------
# MAIN INTERFACE LAYOUT
# -----------------------------------------------------------------
if "file_path" in st.session_state:
    
    # FIX: Render compiled StateGraph in a collapsible section with updated layout parameter
    with st.expander("📊 View Active LangGraph Orchestration Topology", expanded=False):
        st.write("This is the active Compiled StateGraph topology executing your compliance logic:")
        try:
            graph_img_bytes = app.get_graph().draw_mermaid_png()
            # Replaced deprecated parameter with use_container_width
            st.image(graph_img_bytes, use_container_width=True)
        except Exception as e:
            st.caption("Graph visualization unavailable (ensure pygraphviz/mermaid dependencies are configured).")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("System Execution Status")
        
        # Trigger workflow execution stream
        if st.button("🚀 Execute Audit Pipeline", disabled=st.session_state.graph_paused):
            initial_state = {
                "file_path": st.session_state.file_path,
                "file_format": st.session_state.file_format,
                "validation_errors": [],
                "audit_log": []
            }
            
            # Procedural pipeline log progress updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting structured layouts via Docling & GPT-4o...")
            progress_bar.progress(25)
            
            for event in app.stream(initial_state, config):
                if "match_po" in event:
                    status_text.text("PO/GRN context documents successfully loaded from ChromaDB...")
                    progress_bar.progress(60)
                if "validate" in event:
                    status_text.text("Cross-referencing 3-way match validation logic parameters...")
                    progress_bar.progress(90)
            
            progress_bar.progress(100)
            status_text.text("Execution milestone reached.")
            st.rerun()

        # Check framework checkpoint state for interrupts
        state_info = app.get_state(config)
        
        if state_info.next:
            st.session_state.graph_paused = True
            st.warning(f"🛑 Native Graph Breakpoint Encountered. Suspended before node: `{state_info.next[0]}`")
            st.info("Reason: Invoice requires explicit manual oversight (e.g., Value exceeds $50,000 threshold limit).")
            
            if st.button("👤 Approve & Resume Pipeline Thread", type="primary"):
                with st.spinner("Injecting manual approval token and resuming execution thread..."):
                    for event in app.stream(None, config):
                        pass
                st.session_state.graph_paused = False
                st.rerun()
        
        # Display validation summaries once processing finishes
        if state_info.values and not state_info.next:
            values = state_info.values
            decision = values.get("routing_decision", "Needs Review")
            errors = values.get("validation_errors", [])
            
            if decision == "Matched":
                st.success(f"### Status: {decision.upper()} (Processed for Payment)")
            elif decision == "Exception":
                st.error(f"### Status: {decision.upper()} (Rejected / Blocked)")
            else:
                st.warning(f"### Status: {decision.upper()} (Requires Action)")
                
            if errors:
                st.markdown("#### Flagged Compliance Discrepancies:")
                for err in errors:
                    st.markdown(f"- ⚠️ `{err}`")
                    
            if values.get("audit_log"):
                with st.expander("Chronological Audit Log Summary", expanded=False):
                    for log in values.get("audit_log", []):
                        st.text(log)

    with col2:
        st.subheader("Isolated 3-Way Match Verification Components")
        state_info = app.get_state(config)
        
        if state_info.values:
            values = state_info.values
            invoice_data = values.get("extracted_invoice")
            po_context = values.get("matched_po_context")
            grn_context = values.get("matched_grn_context")
            
            # ---------------------------------------------------------
            # COMPONENT 1: INVOICE EXTRACTION DETAILS
            # ---------------------------------------------------------
            if invoice_data:
                st.markdown("### 📄 1. Vendor Invoice Details")
                
                meta_md = f"""
| Attribute | Extracted Value |
| :--- | :--- |
| **Invoice ID** | {invoice_data.get('invoice_id')} |
| **Vendor Name** | {invoice_data.get('vendor_name')} |
| **PO Reference** | {invoice_data.get('po_number')} |
| **Stated Total** | ${invoice_data.get('invoice_total')} ({invoice_data.get('currency', 'USD')}) |
"""
                st.markdown(meta_md)
                
                line_items = invoice_data.get('line_items', [])
                if line_items:
                    st.markdown("#### Itemized Line Entries")
                    df_items = pd.DataFrame(line_items)
                    df_items.columns = ["Item Description", "Quantity", "Unit Price", "Total Price"]
                    st.markdown(df_items.to_markdown(index=False))
            
            # ---------------------------------------------------------
            # COMPONENT 2: PURCHASE ORDER CONTEXT DATA
            # ---------------------------------------------------------
            st.markdown("---")
            st.markdown("### 📜 2. Contracted Purchase Order Details")
            if po_context:
                st.info(f"**Associated Document Source Identity:** `{values.get('matched_po_id')}.md`")
                st.markdown(po_context)
            else:
                st.caption("No Purchase Order context matching text fragments currently localized.")

            # ---------------------------------------------------------
            # COMPONENT 3: GOODS RECEIPT NOTE CONTEXT DATA
            # ---------------------------------------------------------
            st.markdown("---")
            st.markdown("### 📦 3. Warehouse Goods Receipt Details (GRN)")
            if grn_context:
                st.markdown(grn_context)
            else:
                st.caption("No matching Goods Receipt Note text fragments currently localized.")
else:
    st.info("👈 Open the left sidebar and upload an invoice to test the autonomous processing workflow.")