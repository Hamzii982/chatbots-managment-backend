from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Literal
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/ai", tags=["AI Assistance"])

# Defense-in-depth cap: if a client bug ever causes an unbounded tool-calling
# loop, refuse before the history grows unreasonably large rather than
# silently paying for it.
MAX_HISTORY_MESSAGES = 40


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Toolcall / Request / Response schemas
# ---------------------------------------------------------------------------

class ToolCallItem(BaseModel):
    """A single tool invocation requested by the model, or being replayed
    back into history on a follow-up turn."""
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MessageHistory(BaseModel):
    role: str = Field(..., description="'human' | 'ai' | 'system' | 'tool'")
    content: str = ""
    # Only set when role == 'ai' and that turn requested tool calls. Needed so
    # the follow-up turn can be reconstructed correctly (OpenAI requires every
    # AI tool_calls message to be immediately followed by matching tool
    # results).
    tool_calls: list[ToolCallItem] | None = None
    # Only set when role == 'tool'. Must match the id of the tool_calls entry
    # it's answering.
    tool_call_id: str | None = None


class AIAssistRequest(BaseModel):
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="Instructions that shape the model's behaviour for this call.",
    )
    context: str | None = Field(
        default=None,
        description="Extra background information injected before the user message.",
    )
    message: str = Field(..., description="The current user message / query.")
    history: list[MessageHistory] = Field(
        default_factory=list,
        description="Previous turns in the conversation.",
    )
    output_structure: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional JSON schema describing the desired response shape. "
            "Ignored when `tools` is provided — mutually exclusive."
        ),
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional list of OpenAI-format tool definitions "
            '(e.g. {"type": "function", "function": {"name", "description", "parameters"}}). '
            "When provided, the model may respond with tool_calls instead of a final answer."
        ),
    )
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class AIAssistResponse(BaseModel):
    result: Any = Field(default=None, description="Parsed output – JSON object or plain string.")
    raw_text: str | None = Field(default=None, description="Raw text returned by the model, when it produced a final answer.")
    tool_calls: list[ToolCallItem] | None = Field(
        default=None,
        description="Populated instead of raw_text/result when the model wants to call tool(s) before answering.",
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_langchain_history(history: list[MessageHistory]) -> list:
    """Convert our MessageHistory objects to LangChain message objects."""
    messages = []
    for item in history:
        role = item.role.lower()

        if role == "human":
            messages.append(HumanMessage(content=item.content))

        elif role == "system":
            messages.append(SystemMessage(content=item.content))

        elif role == "ai":
            if item.tool_calls:
                # Reconstruct the exact tool_calls this AI turn made, so the
                # follow-up ToolMessages below line up correctly for OpenAI.
                lc_tool_calls = [
                    {
                        "name": tc.name,
                        "args": tc.arguments,
                        "id": tc.id,
                        "type": "tool_call",
                    }
                    for tc in item.tool_calls
                ]
                messages.append(AIMessage(content=item.content, tool_calls=lc_tool_calls))
            else:
                messages.append(AIMessage(content=item.content))

        elif role == "tool":
            if not item.tool_call_id:
                raise ValueError("History item with role 'tool' must include tool_call_id.")
            messages.append(ToolMessage(content=item.content, tool_call_id=item.tool_call_id))

        else:
            raise ValueError(f"Unknown role '{item.role}'. Must be human, ai, system, or tool.")

    return messages


def _build_system_text(
    system_prompt: str,
    context: str | None,
    output_structure: dict[str, Any] | None,
) -> str:
    """Compose the full system message from the individual pieces."""
    parts = [system_prompt]

    if context:
        parts.append(f"\n\n## Context\n{context}")

    if output_structure:
        schema_json = json.dumps(output_structure, indent=2)
        parts.append(
            f"\n\n## Output Format\n"
            f"You MUST respond with a valid JSON object that matches this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Return ONLY the JSON object – no markdown fences, no explanation."
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/assist", response_model=AIAssistResponse)
async def ai_assist(request: AIAssistRequest) -> AIAssistResponse:
    """
    General-purpose AI assistance endpoint.

    Two mutually exclusive modes:
    - `tools` provided → tool-calling mode. Returns either `tool_calls`
        (model wants data before answering) or a plain-text final answer.
    - `output_structure` provided (or neither) → existing structured/plain
        single-shot behaviour, unchanged.
    """
    if request.tools and request.output_structure:
        raise HTTPException(
            status_code=422,
            detail="`tools` and `output_structure` are mutually exclusive in a single call.",
        )

    if len(request.history) > MAX_HISTORY_MESSAGES:
        raise HTTPException(
            status_code=422,
            detail=f"History exceeds {MAX_HISTORY_MESSAGES} messages — refusing, possible runaway tool-calling loop.",
        )

    try:
        llm = ChatOpenAI(model=request.model, temperature=request.temperature)

        system_text = _build_system_text(
            request.system_prompt, request.context, request.output_structure
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{message}"),
            ]
        )

        lc_history = _build_langchain_history(request.history)

        # -------------------------------------------------------------
        # Tool-calling mode
        # -------------------------------------------------------------
        if request.tools:
            llm_with_tools = llm.bind_tools(request.tools)
            chain = prompt | llm_with_tools

            ai_message: AIMessage = await chain.ainvoke(
                {
                    "system_text": system_text,
                    "history": lc_history,
                    "message": request.message,
                }
            )

            if ai_message.tool_calls:
                tool_calls = [
                    ToolCallItem(id=tc["id"], name=tc["name"], arguments=tc["args"])
                    for tc in ai_message.tool_calls
                ]
                return AIAssistResponse(result=None, raw_text=None, tool_calls=tool_calls)

            # No tool call requested — model answered directly.
            return AIAssistResponse(result=ai_message.content, raw_text=ai_message.content, tool_calls=None)

        # -------------------------------------------------------------
        # Existing non-tool-calling behaviour (unchanged)
        # -------------------------------------------------------------
        if request.output_structure:
            parser = JsonOutputParser()
        else:
            parser = StrOutputParser()

        chain = prompt | llm | parser

        result = await chain.ainvoke(
            {
                "system_text": system_text,
                "history": lc_history,
                "message": request.message,
            }
        )

        raw_chain = prompt | llm | StrOutputParser()
        raw_text = await raw_chain.ainvoke(
            {
                "system_text": system_text,
                "history": lc_history,
                "message": request.message,
            }
        )

        return AIAssistResponse(result=result, raw_text=raw_text)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned invalid JSON: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI chain error: {exc}") from exc