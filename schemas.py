from typing import List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. LLM EXTRACTION SCHEMA (Kept as Pydantic for OpenAI)
# ---------------------------------------------------------
class InvoiceLineItem(BaseModel):
    item_description: str = Field(description="The description or name of the product or service.")
    quantity: float = Field(description="The quantity of the item billed.")
    unit_price: float = Field(description="The unit price of the individual item.")
    total_price: float = Field(description="The total calculated price for this line item (qty * unit_price).")

class ExtractedInvoice(BaseModel):
    invoice_id: str = Field(description="The unique identifier or invoice number found on the document.")
    vendor_name: str = Field(description="The name of the company issuing the invoice.")
    po_number: Optional[str] = Field(None, description="The purchase order (PO) number explicitly referenced.")
    line_items: List[InvoiceLineItem] = Field(description="The list of all itemized lines.")
    invoice_total: float = Field(description="The final total invoice balance summary.")
    currency: str = Field("USD", description="The standard currency abbreviation.")

# ---------------------------------------------------------
# 2. THE LANGGRAPH STATE (Refactored to Standard TypedDict)
# ---------------------------------------------------------
class AgentState(TypedDict):
    # Inputs & File Metadata
    file_path: str
    file_format: Literal["pdf", "docx", "png", "jpg"]
    
    # Processed Extraction Artifacts
    raw_text: Optional[str]
    extracted_invoice: Optional[ExtractedInvoice]
    
    # RAG Matching Context
    matched_po_id: Optional[str]
    matched_po_context: Optional[str]
    matched_grn_context: Optional[str]
    
    # Validation & Decision Metrics
    is_duplicate: bool
    validation_errors: List[str]
    
    # Routing Outcome
    routing_decision: Optional[Literal["Matched", "Exception", "Needs Review"]]
    audit_log: List[str]