def generate_mock_response(chatbot):
    return f"[MOCK] Chatbot '{chatbot.name}' responding with system prompt: {chatbot.system_prompt[:50]}"


from requests import Session

from app.services.langchain_rag import answer_with_rag


def generate_rag_response(chatbot, db: Session, message: str, sources: list[str] | None = None, chat_history: list = None) -> dict:
    """Simulate a RAG response for a single selected chatbot by combining system prompt,
    retrieved snippets, and the user message.

    Replace retrieval and LLM call with real integrations when ready.
    """
    # Prefer to use a real RAG pipeline via LangChain
    try:
        out = answer_with_rag(chatbot, message, db, chat_history=chat_history)
        return out
    except Exception as e:
        print(f"Error occurred while generating RAG response: {e}")
        # fallback to lightweight simulation if LangChain call fails
        if sources is None:
            sources = []
            if chatbot.short_term_memory:
                sources.append(f"Short-term memory snippet for {chatbot.name} (chunk_size={chatbot.chunk_size})")
            if chatbot.long_term_memory:
                sources.append(f"Long-term memory snippet for {chatbot.name}")
            sources.append(chatbot.system_prompt[:200])

        combined_context = "\n\n".join(sources) + "\n\n" + message
        response_text = (
            f"[RAG:{chatbot.retriever_type}] Response from '{chatbot.name}': "
            f"Answer to '{message}' using {len(sources)} sources."
        )
        return {"response": response_text, "sources": sources, "context": combined_context}