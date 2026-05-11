import time

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File, Form, status
from typing import List
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.chatbot import Chatbot
from app.models.chatbot_documents import ChatbotDocument
from app.models.model_config import ModelConfig
from app.models.message import Message
from app.models.thread import Thread
from app.schemas.chatbot import ChatbotBase, ChatbotCreate, ChatbotResponse, ChatRequest, ChatResponse
from app.services.chatbot_service import generate_mock_response, generate_rag_response
from app.services.langchain_rag import chunk_text, extract_pdf_text, reset_vectorstore, extract_md_text
from app.core.config import settings
from fastapi.responses import StreamingResponse
import json
import app.graph.instance as graph_instance

import os
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbots", tags=["Chatbots"])

UPLOAD_DIR = "uploads/chatbots"
VECTOR_ROOT = ".vectorstores"
MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = set(ext.lower() for ext in settings.ALLOWED_FILE_EXTENSIONS)
 
CONTENT_TYPE_MAP = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/", response_model=ChatbotResponse)
async def create_chatbot(
    data: ChatbotBase,
    db: Session = Depends(get_db),
):
    # 1. Create chatbot
    chatbot = Chatbot(
        name=data.name,
        system_prompt=data.system_prompt,
        model_id=data.model_id,
        retriever_type=data.retriever_type,
        chunk_size=data.chunk_size,
        chunk_overlap=data.chunk_overlap,
        reranker_type=data.reranker_type,
        top_k=data.top_k,
        short_term_memory=data.short_term_memory,
        long_term_memory=data.long_term_memory,
        is_active=data.is_active,
    )
    db.add(chatbot)
    db.commit()
    db.refresh(chatbot)

    return chatbot

@router.put("/{chatbot_id}", response_model=ChatbotResponse)
async def update_chatbot(
    chatbot_id: int,  # Taken from the URL path
    data: ChatbotBase,
    db: Session = Depends(get_db),
):
    # 1. Fetch existing chatbot
    chatbot = db.query(Chatbot).filter(Chatbot.id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    for key, value in data.model_dump().items():
        setattr(chatbot, key, value)

    db.add(chatbot)
    db.commit()
    db.refresh(chatbot)

    return chatbot

@router.delete("/{chatbot_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chatbot(chatbot_id: int, db: Session = Depends(get_db)):
    """
    Deletes a chatbot and all its associated data (cascading).
    """
    # 1. Fetch the chatbot
    chatbot = db.query(Chatbot).filter(Chatbot.id == chatbot_id).first()
    
    if not chatbot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Chatbot not found"
        )

    try:
        # 2. Delete the chatbot 
        # Note: If your SQLAlchemy relationship has cascade="all, delete", 
        # it will automatically remove messages and documents.
        db.delete(chatbot)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting chatbot: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to delete chatbot"
        )

    # 3. Return 204 No Content (standard for successful deletes)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.get("/", response_model=list[ChatbotResponse])
def get_chatbots(db: Session = Depends(get_db)):
    return db.query(Chatbot).all()

@router.delete("/{chatbot_id}/reset")
def reset_chatbot_history(chatbot_id: int, db: Session = Depends(get_db)):
    """
    Deletes all messages associated with a specific chatbot_id.
    """
    # 1. Check if the chatbot exists first (Optional but recommended)
    chatbot = db.query(Chatbot).filter(Chatbot.id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Chatbot not found"
        )

    # 2. Delete all messages for this chatbot
    try:
        db.query(Message).filter(Message.chatbot_id == chatbot_id).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to reset history: {str(e)}"
        )

    return {"message": f"Successfully cleared history for chatbot '{chatbot.name}'."}

@router.post("/{chatbot_id}/chat")
async def chat(chatbot_id: int, request: ChatRequest, db: Session = Depends(get_db)):
    try:
        chatbot = db.query(Chatbot).get(chatbot_id)
        if not chatbot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chatbot not found"
            )

        # If chatbot is not active, return error
        if not chatbot.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chatbot is not active"
            )

        # Route by retriever type
        rt = (chatbot.retriever_type or "").lower()
        if rt == "mock":
            return {"response": generate_mock_response(chatbot)}

        # treat 'vector' or 'hybrid' retriever types
        if "vector" in rt or "hybrid" in rt:
            chat_history = []
            thread_id = None
            
            # ── NO MEMORY (playground override) ──────────────────────────
            if request.disable_memory:
                chat_history = [("human", request.message)]
            
            # ── LONG TERM MEMORY ──────────────────────────────────────────
            elif chatbot.long_term_memory:
                thread_id = request.thread_id

                # Create new thread if none provided
                if not thread_id:
                    thread = Thread(chatbot_id=chatbot_id, title=request.message[:60])
                    db.add(thread)
                    db.commit()
                    db.refresh(thread)
                    thread_id = thread.id

                # Save user message
                db.add(Message(chatbot_id=chatbot_id, thread_id=thread_id, content=request.message, is_user=True))
                db.commit()

                # Build full history for this thread
                past = db.query(Message)\
                    .filter(Message.chatbot_id == chatbot_id, Message.thread_id == thread_id)\
                    .order_by(Message.created_at.asc()).limit(50).all()
                chat_history = [("human" if m.is_user else "ai", m.content) for m in past]
            
            # ── SHORT TERM MEMORY ─────────────────────────────────────────
            elif chatbot.short_term_memory:
                # Always use the latest thread, create one if none exists
                latest_thread = db.query(Thread)\
                    .filter(Thread.chatbot_id == chatbot_id)\
                    .order_by(Thread.updated_at.desc()).first()

                if not latest_thread:
                    latest_thread = Thread(chatbot_id=chatbot_id, title=request.message[:60])
                    db.add(latest_thread)
                    db.commit()
                    db.refresh(latest_thread)

                thread_id = latest_thread.id
                
                # SAVE USER MESSAGE TO DB
                user_msg = Message(chatbot_id=chatbot_id, thread_id=thread_id, content=request.message, is_user=True)
                db.add(user_msg)
                db.commit()
                
                # Fetch chat history
                past_messages = db.query(Message).filter(Message.chatbot_id == chatbot_id, Message.thread_id == thread_id)\
                                .order_by(Message.created_at.asc())\
                                .limit(50).all()
                
                # Reverse to chronological order
                for msg in past_messages:
                    chat_history.append(("human" if msg.is_user else "ai", msg.content))
                    
            else:
                chat_history = [("human", request.message)]
                    
            # Pass DB session in config so nodes can use it
            config = {
                "configurable": {
                    "thread_id": str(thread_id) if thread_id else str(chatbot_id), 
                    "db": db 
                }
            }
            
            async def event_generator():
                inputs = {
                    "messages": chat_history, 
                    "chatbot_id": chatbot_id
                }
                full_response = []
                
                # Yield thread_id first so frontend knows which thread this belongs to
                yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"
                
                # Use version="v2" for the most consistent schema
                async for event in graph_instance.app_graph.astream_events(
                    inputs, 
                    config, 
                    version="v2"
                ):
                    kind = event["event"]

                    # Filter for model streaming events
                    if kind == "on_chat_model_stream":
                        content = event["data"].get("chunk", {}).content
                        if content:
                            # Standard SSE format: data: {json}\n\n
                            full_response.append(content)
                            yield f"data: {json.dumps({'content': content})}\n\n"

                # Save bot response only if memory is enabled
                if not request.disable_memory and (chatbot.long_term_memory or chatbot.short_term_memory):
                    db.add(Message(
                        chatbot_id=chatbot_id,
                        thread_id=thread_id,
                        content="".join(full_response),
                        is_user=False
                    ))
                    db.commit()
                
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # fallback to mock
        return {"response": generate_mock_response(chatbot)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/{chatbot_id}/messages")
async def get_messages(chatbot_id: int, db: Session = Depends(get_db)):
    chatbot = db.query(Chatbot).get(chatbot_id)
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    past_messages = (
        db.query(Message)
        .filter(Message.chatbot_id == chatbot_id)
        .order_by(Message.created_at.asc())
        .limit(50)
        .all()
    )
    
    return [
        {
            "role": "user" if msg.is_user else "bot",
            "content": msg.content,
        }
        for msg in past_messages
    ]

@router.get("/{chatbot_id}/info")
def get_chatbot_info(chatbot_id: int, db: Session = Depends(get_db)):
    chatbot = db.query(Chatbot).get(chatbot_id)
    if not chatbot:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "id": chatbot.id,
        "name": chatbot.name,
        "long_term_memory": chatbot.long_term_memory,
        "short_term_memory": chatbot.short_term_memory,
    }
    
@router.get("/{chatbot_id}/threads/latest/messages")
def get_latest_thread_messages(chatbot_id: int, db: Session = Depends(get_db)):
    thread = db.query(Thread)\
        .filter(Thread.chatbot_id == chatbot_id)\
        .order_by(Thread.updated_at.desc()).first()
    if not thread:
        return []
    return db.query(Message)\
        .filter(Message.thread_id == thread.id)\
        .order_by(Message.created_at.asc()).all()
        
@router.get("/activated-rag", response_model=list[ChatbotResponse])
def get_activated_rag_chatbots(db: Session = Depends(get_db)):
    # Return chatbots that are active and have a non-mock retriever
    return db.query(Chatbot).filter(Chatbot.is_active == True).filter(Chatbot.retriever_type != "mock").all()

@router.post("/{chatbot_id}/upload-pdfs")
def upload_pdfs(
    chatbot_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload PDF files to a chatbot with security validation.
    Validates file size and extension before saving.
    """
    try:
        chatbot = db.query(Chatbot).filter(Chatbot.id == chatbot_id).first()
        if not chatbot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chatbot not found"
            )

        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )

        saved = []
        errors = []

        for file in files:
            try:
                # Validate file extension
                file_ext = os.path.splitext(file.filename)[1].lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    errors.append(f"{file.filename}: Invalid file type. Only {ALLOWED_EXTENSIONS} allowed.")
                    continue

                # Read file content to check size
                file_content = file.file.read()
                file_size = len(file_content)
                
                if file_size > MAX_UPLOAD_SIZE_BYTES:
                    errors.append(
                        f"{file.filename}: File too large. Max size is {settings.MAX_UPLOAD_SIZE_MB}MB."
                    )
                    continue
                
                if file_size == 0:
                    errors.append(f"{file.filename}: Empty file.")
                    continue

                # Derive content type from extension — don't trust client-supplied value
                content_type = CONTENT_TYPE_MAP[file_ext]
                
                # Save file with sanitized name
                file_id = str(uuid.uuid4())
                path = os.path.join(UPLOAD_DIR, f"{file_id}_{os.path.basename(file.filename)}")

                with open(path, "wb") as f:
                    f.write(file_content)

                db_doc = ChatbotDocument(
                    chatbot_id=chatbot_id,
                    filename=file.filename,
                    file_path=path,
                    content_type=content_type
                )

                db.add(db_doc)
                saved.append(db_doc)
                logger.info(f"Uploaded file: {file.filename} for chatbot {chatbot_id}")

            except Exception as e:
                logger.error(f"Error uploading {file.filename}: {str(e)}")
                errors.append(f"{file.filename}: {str(e)}")
                continue

        # Mark chatbot as needing re-vectorization
        if saved:
            chatbot.is_vectorized = False
            db.commit()

        return {
            "message": "Upload completed",
            "uploaded": len(saved),
            "errors": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed"
        )
    
@router.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(ChatbotDocument).filter(ChatbotDocument.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # 1. Remove from disk
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    
    # 2. Mark chatbot as needing re-vectorization
    chatbot = db.query(Chatbot).filter(Chatbot.id == doc.chatbot_id).first()
    if chatbot:
        chatbot.is_vectorized = False
        
    # 3. Remove from DB
    db.delete(doc)
    db.commit()
    return {"message": "deleted"}

@router.get("/{chatbot_id}/documents")
def get_chatbot_documents(chatbot_id: int, db: Session = Depends(get_db)):
    return db.query(ChatbotDocument).filter(ChatbotDocument.chatbot_id == chatbot_id).all()

@router.post("/{chatbot_id}/vectorize")
def vectorize_chatbot(chatbot_id: int, db: Session = Depends(get_db)):
    """
    Vectorize documents for a chatbot using configured embeddings.
    """
    try:
        chatbot = db.query(Chatbot).filter(Chatbot.id == chatbot_id).first()
        if not chatbot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chatbot not found"
            )

        docs = db.query(ChatbotDocument).filter(
            ChatbotDocument.chatbot_id == chatbot_id
        ).all()

        if not docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents found for this chatbot"
            )
        
        all_chunks = []
        failed_docs = []

        for doc in docs:
            try:
                ext = os.path.splitext(doc.file_path)[1].lower()
 
                if ext == ".pdf":
                    text = extract_pdf_text(doc.file_path)
                elif ext == ".md":
                    text = extract_md_text(doc.file_path)
                else:
                    logger.warning(f"Unsupported file type: {doc.file_path}")
                    failed_docs.append(doc.filename)
                    continue

                chunks = chunk_text(
                    text,
                    chatbot.chunk_size,
                    chatbot.chunk_overlap
                )
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing document {doc.filename}: {str(e)}")
                failed_docs.append(doc.filename)
                continue
        
        if not all_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text chunks extracted from documents"
            )
        
        model_cfg = db.query(ModelConfig).get(chatbot.model_id)
        if not model_cfg or not model_cfg.api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chatbot model configuration missing API key"
            )

        # Set API key for embeddings
        os.environ["OPENAI_API_KEY"] = model_cfg.api_key
        
        try:
            vect = reset_vectorstore(chatbot_id)
            # 2. Process in batches to avoid OpenAI 429 Rate Limits
            batch_size = 50 
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                vect.add_texts(batch)
                
                # Small sleep to let the Rate Limit bucket refill
                # Adjust based on your OpenAI Tier
                time.sleep(1)
        except Exception as e:
            logger.error(f"Vectorization error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to vectorize documents"
            )
        
        chatbot.is_vectorized = True
        db.commit()

        return {
            "status": "success",
            "chunks": len(all_chunks),
            "documents_processed": len(docs) - len(failed_docs),
            "documents_failed": len(failed_docs),
            "failed_documents": failed_docs if failed_docs else None,
            "message": f"Vectorization completed for chatbot {chatbot_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in vectorize: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vectorization failed"
        )
        

# --- Thread Endpoints ---

@router.post("/{chatbot_id}/threads")
def create_thread(chatbot_id: int, db: Session = Depends(get_db)):
    thread = Thread(chatbot_id=chatbot_id, title="New Chat")
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread

@router.get("/{chatbot_id}/threads")
def list_threads(chatbot_id: int, db: Session = Depends(get_db)):
    return db.query(Thread)\
        .filter(Thread.chatbot_id == chatbot_id)\
        .order_by(Thread.updated_at.desc())\
        .all()

@router.get("/{chatbot_id}/threads/{thread_id}/messages")
def get_thread_messages(chatbot_id: int, thread_id: int, db: Session = Depends(get_db)):
    return db.query(Message)\
        .filter(Message.chatbot_id == chatbot_id, Message.thread_id == thread_id)\
        .order_by(Message.created_at.asc())\
        .all()

@router.delete("/{chatbot_id}/threads/{thread_id}")
def delete_thread(chatbot_id: int, thread_id: int, db: Session = Depends(get_db)):
    db.query(Message).filter(Message.thread_id == thread_id).delete()
    db.query(Thread).filter(Thread.id == thread_id).delete()
    db.commit()
    return {"ok": True}