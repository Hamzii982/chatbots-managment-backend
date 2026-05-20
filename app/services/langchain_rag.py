import gc
import os
import shutil
from typing import List

from fastapi import logger
from sqlalchemy.orm import Session
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from app.retrievers.factory import get_retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader

from app.models.model_config import ModelConfig
from app.schemas import chatbot
from app.rerankers.factory import get_reranker


VSTORE_ROOT = ".vectorstores"

def get_path(chatbot_id: int):
    path = os.path.join(VSTORE_ROOT, f"chatbot_{chatbot_id}")
    os.makedirs(path, exist_ok=True)
    return path
    
def reset_vectorstore(chatbot_id: int):
    path = get_path(chatbot_id)
    
    # This helps release file locks
    try:
        gc.collect() 
    except ImportError:
        pass

    # optional hard reset (safe rebuild)
    if os.path.exists(path):
        for i in range(3):
            try:
                shutil.rmtree(path)
                break
            except PermissionError:
                # If shutil fails, the files are definitely still locked by the app
                print(f"Files in {path} are locked. Ensure the VectorStore is closed.")
                # Last ditch effort: wait a second and try again
                import time
                time.sleep(2)

    # Ensure directory exists again
    os.makedirs(path, exist_ok=True)

    return load_vectorstore(chatbot_id)

def load_vectorstore(chatbot_id: int):
    embeddings = OpenAIEmbeddings()

    return Chroma(
        persist_directory=os.path.join(VSTORE_ROOT, f"chatbot_{chatbot_id}"),
        embedding_function=embeddings
    )

def _get_persist_dir(chatbot_id: int) -> str:
    d = os.path.join(VSTORE_ROOT, f"chatbot_{chatbot_id}")
    os.makedirs(d, exist_ok=True)
    return d

def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def extract_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)

        text_parts = []

        for page in reader.pages:
            try:
                text = page.extract_text()
                # HARD GUARD (IMPORTANT)
                if not text:
                    continue

                # filter binary garbage
                if any(ord(c) < 9 for c in text[:50]):
                    continue

                text_parts.append(text)
            except Exception:
                # skip broken pages instead of crashing
                continue
        
        result = "\n".join(text_parts)

        if not result.strip():
            raise ValueError(f"Empty or unreadable PDF: {path}")

        return result

    except Exception as e:
        raise ValueError(f"Failed to parse PDF {path}: {str(e)}")
    
def extract_md_text(file_path: str) -> str:
    """Read a markdown file and return its raw text content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def build_or_get_vectorstore(chatbot_id: int, texts: List[str], embeddings=None) -> Chroma:
    """Create or load a Chroma vectorstore for the chatbot.

    Texts is a list of strings used to populate the index if it's empty.
    """
    persist_dir = _get_persist_dir(chatbot_id)
    if embeddings is None:
        embeddings = OpenAIEmbeddings()

    # Try to load existing collection
    try:
        vect = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        # if empty, add texts
        if vect._collection.count() == 0 and texts:
            vect.add_texts(texts)
        return vect
    except Exception:
        # create new
        vect = Chroma.from_texts(texts or [""], embedding=embeddings, persist_directory=persist_dir)
        vect.persist()
        return vect

def answer_with_rag(chatbot, query: str, db: Session, chat_history: list = None) -> dict:
    """Run a simple RAG retrieval+LLM pipeline using LangChain and Chroma.

    - Populates the vectorstore with the chatbot.system_prompt as a minimal doc if store is empty.
    - Returns {'response': str, 'sources': [str,...]}.
    """
    model_cfg = db.query(ModelConfig).get(chatbot.model_id)
    temperature = model_cfg.temperature if model_cfg else 0.7
    api_key = model_cfg.api_key if model_cfg else None
    model_name = model_cfg.model_name if model_cfg else "gpt-3.5-turbo"
    if not api_key:
        raise ValueError("Chatbot has no OpenAI API key configured")

    # --- Dynamic LLM Loading ---
    provider = model_cfg.provider.lower()
    
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
        llm = ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    # ---------------------------
    
    # Prepare embeddings and vectorstore
    vect = load_vectorstore(chatbot.id)
    retriever = get_retriever(chatbot, vect, db)
    docs = retriever.retrieve(query)
    reranker = get_reranker(chatbot.reranker_type)
    # retriever = vect.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={"k": 5, "fetch_k": 20}
    # )
    
    if reranker:
        docs = reranker.rerank(query, docs)

    # limit final docs
    docs = docs[:chatbot.top_k]
    
    # docs = retriever.invoke(query)
    context = "\n\n".join([getattr(d, "page_content") for d in docs])

    # llm = ChatOpenAI(temperature=temperature)
    
    # 1. Build the prompt structure dynamically
    messages = [("system", chatbot.system_prompt)]
    
    # 2. Add history if it exists and memory is enabled
    if chatbot.short_term_memory and chat_history:
        messages.extend(chat_history)
    
    # 3. Add context and final question
    messages.append(("system", "Verwenden Sie den folgenden Kontext, um zu antworten:\n\n{context}"))
    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return {"response": response.content, "sources": [d.page_content for d in docs], "context": context}


# def stream_rag_response(chatbot, db: Session, query: str, chat_history: list = None):
    """
    Generator that yields text chunks from the LLM.
    """
    model_cfg = db.query(ModelConfig).get(chatbot.model_id)
    if not model_cfg or not model_cfg.api_key:
        yield "Fehler: API-Key nicht konfiguriert."
        return

    # --- Setup LLM (Same as before) ---
    provider = model_cfg.provider.lower()
    os.environ["OPENAI_API_KEY"] = model_cfg.api_key # or ANTHROPIC_API_KEY
    
    if provider == "openai":
        llm = ChatOpenAI(model=model_cfg.model_name, temperature=model_cfg.temperature, streaming=True)
    elif provider == "anthropic":
        llm = ChatAnthropic(model=model_cfg.model_name, temperature=model_cfg.temperature, streaming=True)
    
    # --- Retrieval (Same as before) ---
    vect = load_vectorstore(chatbot.id)
    retriever = get_retriever(chatbot, vect, db)
    docs = retriever.retrieve(query)[:chatbot.top_k]
    context = "\n\n".join([d.page_content for d in docs])

    # --- Build Prompt ---
    messages = [("system", chatbot.system_prompt)]
    if chatbot.short_term_memory and chat_history:
        messages.extend(chat_history)
    
    messages.append(("system", "Verwenden Sie den folgenden Kontext:\n\n{context}"))
    messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm

    # --- Stream Execution ---
    try:
        for chunk in chain.stream({"context": context, "question": query}):
            # Handle both LangChain string chunks and MessageChunk objects
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            yield content
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"\n[Fehler während der Generierung: {str(e)}]"
