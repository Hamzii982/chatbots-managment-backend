from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # add_messages allows appending new messages rather than overwriting
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    chatbot_id: int