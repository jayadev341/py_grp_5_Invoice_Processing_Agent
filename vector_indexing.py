import os
import json
from pathlib import Path
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# Setup directories for Docling text targets
KNOWLEDGE_DIR = Path("data/knowledge_docs")
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. CONVERT DB JSON TO STRUCTURAL MARKDOWN FOR DOCLING
# ---------------------------------------------------------
print("Preparing database summaries for structural parsing...")

with open("data/pos.json", "r") as f:
    po_db = json.load(f)

with open("data/grns.json", "r") as f:
    grn_db = json.load(f)

file_paths_to_load = []

# Generate PO Markdown Files
for po_id, data in po_db.items():
    file_path = KNOWLEDGE_DIR / f"{po_id}.md"
    content = f"""# Purchase Order Document
- **PO Number**: {po_id}
- **Vendor Name**: {data['vendor']}
- **Line Item Description**: {data['item']}
- **Authorized Quantity**: {data['qty']}
- **Contracted Unit Price**: {data['unit_price']}
- **Total Authorized Amount**: {data['total']}
"""
    file_path.write_text(content)
    file_paths_to_load.append(str(file_path))

# Generate GRN Markdown Files
for grn_id, data in grn_db.items():
    file_path = KNOWLEDGE_DIR / f"{grn_id}.md"
    content = f"""# Goods Receipt Note (GRN)
- **GRN ID**: {grn_id}
- **Referenced PO Number**: {data['po_number']}
- **Received Item Description**: {data['item']}
- **Verified Quantity Received**: {data['qty_received']}
"""
    file_path.write_text(content)
    file_paths_to_load.append(str(file_path))

# ---------------------------------------------------------
# 2. RUN DOCLING NATIVE STRUCTURAL CHUNKING
# ---------------------------------------------------------
print(f"Running Docling extraction over {len(file_paths_to_load)} document files...")

# ExportType.DOC_CHUNKS delegates chunking boundaries natively to Docling
loader = DoclingLoader(
    file_path=file_paths_to_load,
    export_type=ExportType.DOC_CHUNKS
)
chunks = loader.load()

print(f"Docling successfully generated {len(chunks)} contextual chunks.")

# ---------------------------------------------------------
# 3. SANITIZE & FLATTEN METADATA FOR CHROMADB COMPLIANCE
# ---------------------------------------------------------
print("Sanitizing complex structural metadata for ChromaDB compatibility...")

cleaned_chunks = []
for chunk in chunks:
    # langchain_docling houses the docling chunk details inside "dl_meta"
    dl_meta = chunk.metadata.get("dl_meta", {})
    
    # Extract origin and headings from dl_meta (handles both dicts and Pydantic models)
    origin = dl_meta.get("origin") if isinstance(dl_meta, dict) else getattr(dl_meta, "origin", None)
    headings = dl_meta.get("headings") if isinstance(dl_meta, dict) else getattr(dl_meta, "headings", None)
    
    # Extract the filename from the origin block
    filename = "unknown"
    if origin:
        filename = origin.get("filename") if isinstance(origin, dict) else getattr(origin, "filename", "unknown")
        
    # Get the primary heading if available
    primary_heading = "Document Section"
    if headings and len(headings) > 0:
        primary_heading = headings[0]

    # Re-assign absolute flat metadata back to the chunk for ChromaDB compliance
    chunk.metadata = {
        "source": str(filename),
        "filename": str(filename),
        "heading": str(primary_heading)
    }
    cleaned_chunks.append(chunk)

# ---------------------------------------------------------
# 4. EMBED AND STORE IN CHROMADB
# ---------------------------------------------------------
print("Initializing OpenAI Text Embeddings (text-embedding-3-large)...")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

print("Indexing clean chunks into ChromaDB...")
vector_store = Chroma.from_documents(
    documents=cleaned_chunks,
    embedding=embedding_model,
    collection_name="invoice_matching_knowledge",
    persist_directory="./chroma_db"
)