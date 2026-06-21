import os
from app import app

# Ensure API Key is assigned
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please export your OPENAI_API_KEY context token variable.")

def execute_invoice(file_name: str, file_ext: str, thread_id: str):
    print(f"\n🚀 Processing target document: {file_name}...")
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "file_path": f"data/invoices/{file_name}",
        "file_format": file_ext,
        "validation_errors": [],
        "audit_log": []
    }
    
    # Run the streaming thread engine
    for event in app.stream(initial_state, config):
        pass
        
    # Inspect if the engine paused for human review
    state_info = app.get_state(config)
    if state_info.next:
        print(f"🛑 [HITL ALERT] Pipeline paused at checkpoint. Next step: {state_info.next}")
        print("👤 Action: Simulating Manual Analyst Review Override -> Approved.")
        # Resume by streaming None into the designated thread session
        for event in app.stream(None, config):
            pass

if __name__ == "__main__":
    print("=====================================================================")
    print("🔥 STARTING AUTOMATED CAPSTONE CAPABILITY EVALUATION DEMO")
    print("=====================================================================\n")
    
    # Scenario 1: Clean PDF Invoice (Perfect 3-way match)
    execute_invoice("INV-001.pdf", "pdf", "thread_001")
    
    # Scenario 2: Scanned Image Invoice (Requires OCR parsing engine)
    execute_invoice("INV-003.png", "png", "thread_003")
    
    # Scenario 3: Price/Quantity Deviation Mismatch
    execute_invoice("INV-004.pdf", "pdf", "thread_004")
    
    # Scenario 4: Duplicate Invoice Attempt (Re-submitting structural payload data)
    execute_invoice("INV-007.pdf", "pdf", "thread_007")
    
    # Scenario 5: High-Value Threshold Guardrail Review ($75,000 total requiring sign-off)
    execute_invoice("INV-010.pdf", "pdf", "thread_010")
    
    print("=====================================================================")
    print("✅ BATCH PERFORMANCE TEST COMPLETE")
    print("=====================================================================")