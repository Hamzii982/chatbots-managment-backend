from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.graph.nodes import call_model, retrieve_context

def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("model", call_model)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "model")
    workflow.add_edge("model", END)

    return workflow.compile()