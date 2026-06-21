import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from docx import Document
from PIL import Image, ImageDraw, ImageFont

# Create directory structure
os.makedirs("data/invoices", exist_ok=True)

# ---------------------------------------------------------
# 1. GROUND TRUTH DATASETS (To be indexed in Vector Store)
# ---------------------------------------------------------
po_db = {
    "PO-101": {"vendor": "Nexus Tech", "item": "Cloud Licenses", "qty": 10, "unit_price": 500.0, "total": 5000.0},
    "PO-102": {"vendor": "Apex Office", "item": "Ergonomic Chairs", "qty": 5, "unit_price": 300.0, "total": 1500.0},
    "PO-103": {"vendor": "Pixel Corp", "item": "4K Monitors", "qty": 2, "unit_price": 400.0, "total": 800.0},
    "PO-104": {"vendor": "Nexus Tech", "item": "Laptops", "qty": 10, "unit_price": 1000.0, "total": 10000.0},
    "PO-105": {"vendor": "LogiSupply", "item": "Webcams", "qty": 20, "unit_price": 120.0, "total": 2400.0},
    "PO-106": {"vendor": "Apex Office", "item": "Desk Lamps", "qty": 10, "unit_price": 50.0, "total": 500.0},
    "PO-110": {"vendor": "Enterprise Core", "item": "Server Rack", "qty": 1, "unit_price": 75000.0, "total": 75000.0}
}

grn_db = {
    "GRN-101": {"po_number": "PO-101", "item": "Cloud Licenses", "qty_received": 10},
    "GRN-102": {"po_number": "PO-102", "item": "Ergonomic Chairs", "qty_received": 5},
    "GRN-103": {"po_number": "PO-103", "item": "4K Monitors", "qty_received": 2},
    "GRN-104": {"po_number": "PO-104", "item": "Laptops", "qty_received": 10}, # GRN says 10 received
    "GRN-105": {"po_number": "PO-105", "item": "Webcams", "qty_received": 20},
    "GRN-106": {"po_number": "PO-106", "item": "Desk Lamps", "qty_received": 10},
    "GRN-110": {"po_number": "PO-110", "item": "Server Rack", "qty_received": 1}
}

with open("data/pos.json", "w") as f:
    json.dump(po_db, f, indent=4)
with open("data/grns.json", "w") as f:
    json.dump(grn_db, f, indent=4)

# ---------------------------------------------------------
# 2. FILE GENERATION UTILITIES
# ---------------------------------------------------------
def create_pdf(filename, text_lines):
    c = canvas.Canvas(f"data/invoices/{filename}", pagesize=letter)
    y = 750
    for line in text_lines:
        c.drawString(80, y, line)
        y -= 25
    c.save()

def create_docx(filename, text_lines):
    doc = Document()
    for line in text_lines:
        doc.add_paragraph(line)
    doc.save(f"data/invoices/{filename}")

def create_image(filename, text_lines):
    img = Image.new('RGB', (600, 450), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 40
    for line in text_lines:
        d.text((40, y), line, fill=(0, 0, 0))
        y += 30
    img.save(f"data/invoices/{filename}")

# ---------------------------------------------------------
# 3. GENERATE THE 10 INVOICES
# ---------------------------------------------------------

# INV-001: Clean Pass PDF
create_pdf("INV-001.pdf", [
    "INVOICE", "Invoice ID: INV-001", "Vendor: Nexus Tech", 
    "PO Number: PO-101", "Item: Cloud Licenses", "Qty: 10", 
    "Unit Price: 500.00", "Total: 5000.00", "Currency: USD"
])

# INV-002: Clean Pass DOCX
create_docx("INV-002.docx", [
    "INVOICE", "Invoice ID: INV-002", "Vendor: Apex Office", 
    "PO Number: PO-102", "Item: Ergonomic Chairs", "Qty: 5", 
    "Unit Price: 300.00", "Total: 1500.00", "Currency: USD"
])

# INV-003: Clean Pass Image (For OCR testing)
create_image("INV-003.png", [
    "INVOICE", "Invoice ID: INV-003", "Vendor: Pixel Corp", 
    "PO Number: PO-103", "Item: 4K Monitors", "Qty: 2", 
    "Unit Price: 400.00", "Total: 800.00", "Currency: USD"
])

# INV-004: Quantity Mismatch PDF (Invoice requests 15, PO/GRN only verified 10)
create_pdf("INV-004.pdf", [
    "INVOICE", "Invoice ID: INV-004", "Vendor: Nexus Tech", 
    "PO Number: PO-104", "Item: Laptops", "Qty: 15", 
    "Unit Price: 1000.00", "Total: 15000.00", "Currency: USD"
])

# INV-005: Price Mismatch DOCX (Invoice charges 150, PO locks at 120)
create_docx("INV-005.docx", [
    "INVOICE", "Invoice ID: INV-005", "Vendor: LogiSupply", 
    "PO Number: PO-105", "Item: Webcams", "Qty: 20", 
    "Unit Price: 150.00", "Total: 3000.00", "Currency: USD"
])

# INV-006: Broken Math Image (Line total says 1200, but 10 * 50 = 500)
create_image("INV-006.png", [
    "INVOICE", "Invoice ID: INV-006", "Vendor: Apex Office", 
    "PO Number: PO-106", "Item: Desk Lamps", "Qty: 10", 
    "Unit Price: 50.00", "Total: 1200.00", "Currency: USD"
])

# INV-007: Duplicate PDF (Identical data to INV-001, separate ID to test tracking)
create_pdf("INV-007.pdf", [
    "INVOICE", "Invoice ID: INV-001", "Vendor: Nexus Tech", 
    "PO Number: PO-101", "Item: Cloud Licenses", "Qty: 10", 
    "Unit Price: 500.00", "Total: 5000.00", "Currency: USD"
])

# INV-008: Non-existent PO PDF
create_pdf("INV-008.pdf", [
    "INVOICE", "Invoice ID: INV-008", "Vendor: Unknown Corp", 
    "PO Number: PO-999", "Item: Mystery Gadgets", "Qty: 1", 
    "Unit Price: 250.00", "Total: 250.00", "Currency: USD"
])

# INV-009: Missing PO field layout DOCX
create_docx("INV-009.docx", [
    "INVOICE", "Invoice ID: INV-009", "Vendor: Apex Office", 
    "Item: General Office Paper", "Qty: 10", 
    "Unit Price: 15.00", "Total: 150.00", "Currency: USD"
])

# INV-010: Exceeds Auto-Approval Limit ($75,000 vs $50,000 threshold)
create_pdf("INV-010.pdf", [
    "INVOICE", "Invoice ID: INV-010", "Vendor: Enterprise Core", 
    "PO Number: PO-110", "Item: Server Rack", "Qty: 1", 
    "Unit Price: 75000.00", "Total: 75000.00", "Currency: USD"
])

print("Successfully generated ground truth data and 10 mock multi-format invoices!")