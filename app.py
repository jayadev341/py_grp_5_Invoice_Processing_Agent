from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from schemas import AgentState
from nodes import extract_node, match_po_node, validate_node, human_review_node, respond_node

def decide_routing(state: AgentState) -> str:
    """Evaluates the immutable node decision token to select the correct execution track."""
    decision = state.get("routing_decision", "Needs Review")
    
    if decision == "Needs Review":
        return "human_review"  # Route directly to Checkpointer Interrupt Gate
        
    return "respond"  # Route direct paths (Matched / Exception) straight to reporting

# =====================================================================
# INITIALIZE GRAPH & RETRY POLICIES
# =====================================================================
llm_retry_policy = RetryPolicy(
    initial_interval=2.0,
    backoff_factor=2.0,
    max_interval=60.0,
    max_attempts=3,
    retry_on=Exception
)

workflow = StateGraph(AgentState)

# Register nodes with retry capabilities
workflow.add_node("extract", extract_node, retry=llm_retry_policy)
workflow.add_node("match_po", match_po_node, retry=llm_retry_policy)
workflow.add_node("validate", validate_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("respond", respond_node)

# Core linear connections
workflow.set_entry_point("extract")
workflow.add_edge("extract", "match_po")
workflow.add_edge("match_po", "validate")

# Apply conditional edge routing
workflow.add_conditional_edges(
    "validate",
    decide_routing,
    {
        "respond": "respond",
        "human_review": "human_review"
    }
)

workflow.add_edge("human_review", "respond")
workflow.add_edge("respond", END)

memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_review"]
)