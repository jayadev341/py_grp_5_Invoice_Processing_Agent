# **Project 5 — Automated Invoice Processing Agent (Custom Python · LangChain / LangGraph)**

# **1\. Project Overview**

### **Goal:**

Build an agentic invoice-processing system in Python using:

* OpenAI LLMs (GPT-4o / GPT-4.1) for extraction and reasoning  
* OpenAI Embeddings (text-embedding-3-large) for PO/vendor matching  
* ChromaDB / FAISS for vendor & purchase-order knowledge  
* LangChain \+ LangGraph for the agentic workflow

The system must:

* Ingest invoices (PDF, scanned images, DOCX) and extract structured fields  
* Validate invoice data against matching purchase orders (3-way match)( Purchase order,goods receipt note, vendor invoice)  
* Detect exceptions (price/quantity mismatch, duplicate, missing PO)  
* Produce structured, validated output (Pydantic) and route exceptions for human review  
* Include guardrails for safety, reliability, and auditability

# **2\. High-Level Architecture**

### **Components:**

* Ingestion — load PDFs/images/DOCX; OCR for scans (e.g., pytesseract / Azure Document Intelligence)  
* Extraction — LLM extracts invoice fields into a Pydantic schema  
* Knowledge / Vector Store — PO and vendor master indexed in ChromaDB/FAISS  
* LangGraph Agent — nodes: extract → match\_PO → validate → decide → respond  
* Tools — PO lookup, duplicate-check, totals validator/calculator  
* Structured Output \+ Guardrails — Pydantic validation, exception routing  
* Frontend / API — Streamlit or FastAPI( OPTIONAL)

# **3\. Data Model Design (Pydantic \+ Vector Store)**

### **Pydantic: Invoice**

| Field | Type | Description |
| :---- | :---- | :---- |
| **invoice\_id** | str | Invoice number |
| **vendor\_name** | str | Supplier name |
| **po\_number** | str | None | Referenced purchase order |
| **line\_items** | list\[LineItem\] | description, qty, unit\_price, amount |
| **subtotal/tax/total** | float | Monetary fields |
| **currency** | str | ISO currency |
| **status** | enum | Matched / Exception / Needs Review |

### **Vector store metadata**

* doc\_id, po\_number, vendor\_id, source, page

# **4\. Ingestion Pipeline (LangChain)**

### **Step 1 — Load Documents**

Use LangChain document loaders for PDF/DOCX and an OCR step for images. A minimum of 10 sample invoices across formats (PDF, scanned image, DOCX) must be ingested.

### **Step 2 — Preprocessing**

* De-skew/clean OCR text; normalize whitespace; detect layout/tables

### **Step 3 — Field Extraction (LLM)**

* Prompt the LLM to extract into the Invoice Pydantic schema (function/tool calling)

### **Step 4 — Embedding & Indexing**

* Embed vendor master \+ open POs; store in ChromaDB/FAISS for matching

# **5\. Agentic Processing Pipeline (LangGraph)**

### **Graph Logic**

1. extract: parse invoice → Pydantic Invoice  
2. match\_PO: retrieve the best-matching PO from the vector store  
3. validate: 3-way match (invoice vs PO vs receipt) — price, qty, totals  
4. duplicate\_check: detect previously processed invoices  
5. decide: Matched → approve; mismatch → Exception; ambiguous → Needs Review  
6. respond: emit structured result \+ reason; route exceptions to human queue

### **Note: Use a Prompt Template for extraction and a tool schema for PO lookup.**

# **6\. Structured Output Using Pydantic**

All extraction and decisions must be validated against Pydantic models (Invoice, LineItem, MatchResult). Invalid or low-confidence extractions are routed to 'Needs Review'.

# **7\. Safety & Guardrails**

* Temperature 0 for deterministic extraction  
* Refuse/flag low-confidence extractions instead of guessing  
* Validate totals arithmetic (sum of line items \== subtotal)  
* Duplicate-invoice prevention  
* Full audit log of every decision and the evidence used  
* Never auto-approve above a configurable amount threshold without human sign-off

# **8\. End-to-End Agent (LangGraph)**

Assemble the LangGraph StateGraph wiring the nodes above with conditional edges (Matched / Exception / Needs Review), bounded retries, and a human-in-the-loop interrupt for exceptions.

# **9\. Deliverables (Capstone Requirements)**

### **Part A — Engineering**

* Ingestion \+ OCR pipeline for PDF/image/DOCX invoices  
* LLM field extraction into Pydantic schema  
* Vector store of vendor master \+ POs  
* LangGraph agent: extract → match → validate → decide → respond  
* PO-lookup / duplicate-check / totals-validator tools  
* Guardrails \+ audit logging

### **Part B — Analytics**

* Compare extraction accuracy across formats (PDF vs scanned image)  
* Evaluate PO-matching precision@k  
* Measure exception-detection precision/recall on seeded errors  
* Compare prompt/temperature settings on extraction reliability

### **Part C — Final UI**

* Streamlit app to upload an invoice and view the extracted \+ matched result  
* or a Notebook playground with interactive invoice upload

# **10\. Final Submission Requirements**

Participants must submit a presentation containing all results, visuals, and relevant screenshots. Additionally, they must provide a working demo of the code that supports all the following queries/scenarios:

* Clean PDF invoice with matching PO → Matched \+ approved  
* Scanned image invoice (OCR) → extracted \+ matched  
* Price/quantity mismatch vs PO → Exception with reason  
* Duplicate invoice → flagged and blocked  
* Missing PO / above threshold → Needs Review (human-in-the-loop)

The submission must include: source code repository, requirements.txt/environment file, the evaluation notebook with metrics, sample input data, and a live walkthrough.