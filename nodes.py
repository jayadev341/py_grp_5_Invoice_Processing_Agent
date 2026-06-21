# 1. LOAD ENVIRONMENT VARIABLES FIRST
from dotenv import load_dotenv
load_dotenv()

# 2. STANDARD SERVICE IMPORTS
import os
import base64
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from schemas import AgentState, ExtractedInvoice

# =====================================================================
# SHARED CONFIGURATIONS & COMPONENT INITIALIZATION
# =====================================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0.0).with_structured_output(ExtractedInvoice)

shared_pipeline_options = PdfPipelineOptions()
shared_pipeline_options.do_ocr = True
shared_pipeline_options.ocr_options = EasyOcrOptions(lang=["en"], use_gpu=False)

ocr_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=shared_pipeline_options),
        InputFormat.IMAGE: PdfFormatOption(pipeline_options=shared_pipeline_options)
    }
)

PAID_INVOICES_REGISTRY = {"INV-OLD-999"} 

# Helper function for Multimodal Vision encoding
def encode_image(image_path: str) -> str:
    """Encodes a local image file into a base64 string for multimodal LLM consumption."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# =====================================================================
# LANGGRAPH NODE IMPLEMENTATIONS
# =====================================================================

def extract_node(state: AgentState) -> dict:
    """Ingests the document file layout, executing hybrid parsing (Vision for images, Docling for PDFs)."""
    file_path = state["file_path"]
    file_format = state["file_format"].lower()
    logs = state.get("audit_log", []).copy()
    logs.append(f"[extract] Starting ingestion for: {file_path}")
    
    system_prompt = (
        "You are an expert financial audit agent. Analyze the provided invoice document "
        "and carefully extract the target fields into the specified structured format. "
        "Do not compute or fix calculations; extract exactly what is explicitly written."
    )

    # ---------------------------------------------------------
    # PATHWAY A: MULTIMODAL VISION FOR IMAGES (Hyper-Focused Prompt)
    # ---------------------------------------------------------
    if file_format in ["png", "jpg", "jpeg"]:
        logs.append(f"[extract] Image format detected ({file_format}). Activating Multimodal Vision track...")
        base64_image = encode_image(file_path)
        
        # Explicit target breakdown to force model focus on the PO layout line
        vision_instruction = (
            "Examine the invoice image line-by-line and extract the following attributes:\n"
            "- invoice_id\n"
            "- vendor_name\n"
            "- po_number (Look closely for 'PO Number: PO-103' or similar strings and extract the ID)\n"
            "- invoice_total\n"
            "- line_items (quantity, unit_price, total_price, item_description)\n\n"
            "Ensure the 'po_number' field is filled with the exact value found on the image."
        )
        
        structured_result = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": vision_instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{file_format};base64,{base64_image}"}
                    }
                ]
            )
        ])
        raw_markdown = f"### Multimodal Visual Parsing Complete\nDocument processed directly via native vision tracks."
        logs.append(f"[extract] Visual extraction complete for Invoice ID: {structured_result.invoice_id}")

    # ---------------------------------------------------------
    # PATHWAY B: STRUCTURAL DOCLING PARSING FOR PDFs
    # ---------------------------------------------------------
    else:
        logs.append(f"[extract] PDF format detected. Invoking Docling structural parsing layout...")
        loader = DoclingLoader(
            file_path=file_path,
            converter=ocr_converter,
            export_type=ExportType.MARKDOWN
        )
        docs = loader.load()
        raw_markdown = docs[0].page_content if docs else ""
        logs.append(f"[extract] Generated Markdown string ({len(raw_markdown)} characters)")
        
        structured_result = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Invoice Document:\n\n{raw_markdown}"}
        ])
        logs.append(f"[extract] Text extraction complete for Invoice ID: {structured_result.invoice_id}")

    return {
        "raw_text": raw_markdown,
        "extracted_invoice": structured_result.model_dump(),
        "audit_log": logs
    }


def match_po_node(state: AgentState) -> dict:
    """Queries ChromaDB using an isolated Two-Pass RAG flow with strict identity filtering."""
    logs = state.get("audit_log", []).copy()
    invoice = state.get("extracted_invoice")
    
    if not invoice:
        return {"routing_decision": "Needs Review", "validation_errors": ["Missing extraction data"], "audit_log": logs}
    
    po_number = invoice.get("po_number")
    vendor_name = invoice.get("vendor_name")
    
    if po_number == "None" or not po_number:
        po_number = None
        
    logs.append(f"[match_po] Ingested PO token: '{po_number}', Vendor: '{vendor_name}'")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="invoice_matching_knowledge",
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    
    po_hits = []
    grn_hits = []
    
    if po_number:
        logs.append(f"[match_po] Executing isolated target search for PO: '{po_number}'")
        po_candidates = vector_store.similarity_search(f"Purchase Order Document PO Number: {po_number}", k=5)
        grn_candidates = vector_store.similarity_search(f"Goods Receipt Note GRN Referenced PO Number: {po_number}", k=5)
        
        # STRICT FIX: Retain only exact string matches. No loose element array fallbacks.
        po_hits = [doc for doc in po_candidates if po_number in doc.page_content]
        grn_hits = [doc for doc in grn_candidates if po_number in doc.page_content]
    else:
        logs.append(f"[match_po] No PO number on layout. Initiating Two-Pass Fallback Lookup for: '{vendor_name}'")
        po_candidates = vector_store.similarity_search(f"Purchase Order Document Vendor: {vendor_name}", k=5)
        po_hits = po_candidates[:1] if po_candidates else []
        
        if po_hits:
            for line in po_hits[0].page_content.split("\n"):
                if "PO Number" in line:
                    po_number = line.split(":")[-1].strip().replace("**", "").replace(" ", "")
            
            if po_number:
                grn_candidates = vector_store.similarity_search(f"Goods Receipt Note GRN Referenced PO Number: {po_number}", k=5)
                grn_hits = [doc for doc in grn_candidates if po_number in doc.page_content]

    po_context = po_hits[0].page_content if po_hits else None
    grn_context = grn_hits[0].page_content if grn_hits else None
    matched_po_id = po_hits[0].metadata.get("filename", "").replace(".md", "") if po_hits else "NOT_FOUND"

    return {
        "matched_po_id": matched_po_id,
        "matched_po_context": po_context,
        "matched_grn_context": grn_context,
        "audit_log": logs
    }
    
    
def validate_node(state: AgentState) -> dict:
    """Performs full 3-way validation triangle verification with explicit identity compliance tracking."""
    logs = state.get("audit_log", []).copy()
    invoice = state.get("extracted_invoice")
    po_context = state.get("matched_po_context")
    grn_context = state.get("matched_grn_context")
    
    errors = []
    is_duplicate = False
    logs.append("[validate] Running comprehensive compliance guardrails...")

    invoice_id = invoice.get("invoice_id")
    invoice_total = invoice.get("invoice_total", 0.0)
    po_number = invoice.get("po_number")

    if invoice_id in PAID_INVOICES_REGISTRY:
        is_duplicate = True
        errors.append(f"Security Alert: Duplicate Invoice ID detected. '{invoice_id}' has already been processed.")
        return {"validation_errors": errors, "is_duplicate": True, "routing_decision": "Exception", "audit_log": logs}

    # ---------------------------------------------------------
    # LEG A: Context Existence Verification (Catches Unknown Documents)
    # ---------------------------------------------------------
    if not po_context or "NOT_FOUND" in state.get("matched_po_id", ""):
        errors.append(f"Compliance Violation: Stated Purchase Order reference '{po_number}' does not exist in the corporate index.")
    if not grn_context:
        errors.append("Compliance Violation: No corresponding warehouse fulfillment record (GRN) exists for this order sequence.")
        
    if errors:
        logs.append(f"[validate] Reference validation check failed: {errors}")
        return {"validation_errors": errors, "routing_decision": "Exception", "audit_log": logs}

    # ---------------------------------------------------------
    # LEG B: Explicit Document Content String Crosstracking
    # ---------------------------------------------------------
    po_qty, po_price = 0.0, 0.0
    db_po_id = None
    for line in po_context.split("\n"):
        if "PO Number" in line: db_po_id = line.split(":")[-1].strip().replace("**", "")
        if "Authorized Quantity" in line: po_qty = float(line.split(":")[-1].strip())
        if "Contracted Unit Price" in line: po_price = float(line.split(":")[-1].strip())

    # Secondary validation safeguard to confirm the correct matching ID file was parsed
    if po_number and db_po_id and po_number != db_po_id:
        errors.append(f"Security Tampering Alert: Document routing conflict. Invoice references {po_number} but matched {db_po_id}.")
        return {"validation_errors": errors, "routing_decision": "Exception", "audit_log": logs}

    grn_qty = 0.0
    for line in grn_context.split("\n"):
        if "Verified Quantity Received" in line: grn_qty = float(line.split(":")[-1].strip())

    # ---------------------------------------------------------
    # LEG C: Standard Quantitative 3-Way Triad Evaluations
    # ---------------------------------------------------------
    invoice_qty = sum(item.get("quantity", 0.0) for item in invoice.get("line_items", []))
    invoice_max_price = max(item.get("unit_price", 0.0) for item in invoice.get("line_items", [])) if invoice.get("line_items") else 0.0

    if invoice_qty > po_qty:
        errors.append(f"Quantity Mismatch: Invoice Qty {invoice_qty} exceeds PO authorized Qty {po_qty}.")
    if invoice_max_price > po_price:
        errors.append(f"Price Mismatch: Invoice unit price ${invoice_max_price} exceeds contracted PO price ${po_price}.")
    if invoice_qty > grn_qty:
        errors.append(f"Quantity Mismatch: Invoice Qty {invoice_qty} exceeds Warehouse GRN received Qty {grn_qty}.")

    if grn_qty > po_qty:
        errors.append(f"Warehouse Leak Exception: Warehouse received Qty {grn_qty}, which violates PO authorized Qty {po_qty}.")
    elif grn_qty < po_qty:
        logs.append(f"[validate] Note: Short shipment detected. Warehouse received {grn_qty}/{po_qty} authorized units.")

    if errors:
        routing_decision = "Exception"
    elif invoice_total > 50000.0:
        routing_decision = "Needs Review"
    else:
        routing_decision = "Matched"
        PAID_INVOICES_REGISTRY.add(invoice_id)
        logs.append(f"[validate] Invoice {invoice_id} successfully registered in paid history tracker.")

    return {
        "validation_errors": errors,
        "is_duplicate": is_duplicate,
        "routing_decision": routing_decision,
        "audit_log": logs
    }


def human_review_node(state: AgentState) -> dict:
    logs = state.get("audit_log", []).copy()
    logs.append("[human_review] Pipeline suspended natively. Awaiting manual override approval signal...")
    return {"audit_log": logs}


def respond_node(state: AgentState) -> dict:
    logs = state.get("audit_log", []).copy()
    decision = state.get("routing_decision", "Needs Review")
    errors = state.get("validation_errors", [])
    invoice = state.get("extracted_invoice")
    
    logs.append(f"[respond] Operational workflow finalized with status: {decision}")
    
    print("\n" + "="*60)
    print(f"📋 FINAL INVOICE AUDIT REPORT ({decision.upper()})")
    print("="*60)
    if invoice:
        print(f"Invoice ID: {invoice.get('invoice_id')} | Vendor: {invoice.get('vendor_name')} | Total Stated: ${invoice.get('invoice_total')}")
    if errors:
        print("\nIdentified Discrepancies:")
        for err in errors:
            print(f"  ⚠️ {err}")
    print("\nExecution Trail:")
    for log in logs:
        print(f"  {log}")
    print("="*60 + "\n")
    
    return {"audit_log": logs}