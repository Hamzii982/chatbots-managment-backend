import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from app.models.model_config import ModelConfig
from app.models.chatbot import Chatbot
from langchain_core.messages import AIMessageChunk
from app.services.langchain_rag import load_vectorstore, get_retriever # Adjust imports

def retrieve_context(state, config):
    """
    Node 1: Fetch documents based on the last user message.
    """
    # 1. Extract IDs from config/state
    chatbot_id = state.get("chatbot_id")
    db = config.get("configurable", {}).get("db") # Passed from API
    
    # 2. Get the last user message
    user_query = state["messages"][-1].content

    # 3. Existing Retrieval Logic
    chatbot = db.query(Chatbot).get(chatbot_id)
    model_cfg = db.query(ModelConfig).get(chatbot.model_id)
    if not model_cfg or not model_cfg.api_key:
        return

    os.environ["OPENAI_API_KEY"] = model_cfg.api_key # or ANTHROPIC_API_KEY
    
    vect = load_vectorstore(chatbot.id)
    retriever = get_retriever(chatbot, vect, db)
    
    docs = retriever.retrieve(user_query)[:chatbot.top_k]
    context_text = "\n\n".join([d.page_content for d in docs])

    # 4. Update state with context
    return {"context": context_text}


async def call_model(state, config):
    """
    Node 2: Build the prompt and and stream tokens from the LLM.
    Must use chain.astream() so LangGraph can forward individual tokens
    when the graph is invoked with stream_mode="messages".
    """
    chatbot_id = state.get("chatbot_id")
    db = config.get("configurable", {}).get("db")
    context = state.get("context", "")
    
    # 1. Setup Model Config
    chatbot = db.query(Chatbot).get(chatbot_id)
    model_cfg = db.query(ModelConfig).get(chatbot.model_id)
    
    if not model_cfg or not model_cfg.api_key:
        raise ValueError("API-Key nicht konfiguriert.")

    provider = model_cfg.provider.lower()
    
    # 2. Initialize LLM (Streaming is handled by .astream in the API)
    if provider == "openai":
        llm = ChatOpenAI(
            model=model_cfg.model_name, 
            temperature=model_cfg.temperature, 
            api_key=model_cfg.api_key,
            streaming=True
        )
    elif provider == "anthropic":
        llm = ChatAnthropic(
            model=model_cfg.model_name, 
            temperature=model_cfg.temperature, 
            api_key=model_cfg.api_key,
            streaming=True
        )

    # 3. Build Prompt (Mirroring your logic)
    # LangGraph automatically includes chat_history in state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", chatbot.system_prompt),
        ("placeholder", "{messages}"), # This inserts the history
        ("system", "Verwenden Sie den folgenden Kontext:\n\n{context}"),
    ])

    chain = prompt | llm
    
    # ✅ Collect chunks — LangGraph streams them token-by-token via
    # stream_mode="messages" automatically, because streaming=True on the LLM
    chunks = []
    # Directly yield each chunk — do NOT collect into a list first
    final_message = await chain.ainvoke({
        "messages": state["messages"],
        "context": context,
    }, config=config)
        # chunks.append(chunk)
        
    # Return the final assembled message as a state update
    # final_message = sum(chunks[1:], chunks[0])
    return {"messages": [final_message]}