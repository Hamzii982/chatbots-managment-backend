from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/ai", tags=["AI Assistance"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class MessageHistory(BaseModel):
    role: str = Field(..., description="'human' | 'ai' | 'system'")
    content: str


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
            "When provided the model is instructed to return valid JSON "
            "matching this schema."
        ),
    )
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class AIAssistResponse(BaseModel):
    result: Any = Field(..., description="Parsed output – JSON object or plain string.")
    raw_text: str = Field(..., description="Raw text returned by the model.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_langchain_history(history: list[MessageHistory]) -> list:
    """Convert our MessageHistory objects to LangChain message objects."""
    mapping = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
    }
    messages = []
    for item in history:
        cls = mapping.get(item.role.lower())
        if cls is None:
            raise ValueError(f"Unknown role '{item.role}'. Must be human, ai, or system.")
        messages.append(cls(content=item.content))
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

    Accepts a system prompt, optional context, conversation history, a user
    message, and an optional output structure (JSON schema).  Returns the
    model response either as a parsed JSON object (when output_structure is
    supplied) or as a plain string.
    """
    try:
        llm = ChatOpenAI(model=request.model, temperature=request.temperature)

        system_text = _build_system_text(
            request.system_prompt, request.context, request.output_structure
        )

        # Build the prompt template with a slot for history
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{message}"),
            ]
        )

        lc_history = _build_langchain_history(request.history)

        # Choose parser based on whether a structure was requested
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

        # Get raw text for transparency
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