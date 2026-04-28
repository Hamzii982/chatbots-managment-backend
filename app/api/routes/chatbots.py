import time

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File, Form, status
from typing import List
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.chatbot import Chatbot
from app.models.chatbot_documents import ChatbotDocument
from app.models.model_config import ModelConfig
from app.models.message import Message
from app.schemas.chatbot import ChatbotBase, ChatbotCreate, ChatbotResponse, ChatRequest, ChatResponse
from app.services.chatbot_service import generate_mock_response, generate_rag_response
from app.services.langchain_rag import chunk_text, extract_pdf_text, reset_vectorstore, stream_rag_response
from app.core.config import settings
from fastapi.responses import StreamingResponse
import json

import os
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbots", tags=["Chatbots"])

UPLOAD_DIR = "uploads/chatbots"
VECTOR_ROOT = ".vectorstores"
MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = set(ext.lower() for ext in settings.ALLOWED_FILE_EXTENSIONS)

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
def chat(chatbot_id: int, request: ChatRequest, db: Session = Depends(get_db)):
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
            if chatbot.short_term_memory:
                # SAVE USER MESSAGE TO DB
                user_msg = Message(chatbot_id=chatbot_id, content=request.message, is_user=True)
                db.add(user_msg)
                db.commit()
                
                # Fetch chat history
                past_messages = db.query(Message).filter(Message.chatbot_id == chatbot_id)\
                                .order_by(Message.created_at.desc()).limit(50).all()
                
                # Reverse to chronological order
                for msg in reversed(past_messages):
                    chat_history.append(("human" if msg.is_user else "ai", msg.content))
            
            def event_generator():
                full_response = ""
                # Get the stream from our RAG logic
                stream = stream_rag_response(chatbot, db, request.message, chat_history)
                
                for chunk in stream:
                    full_response += chunk
                    # Standard SSE format: data: {...}\n\n
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                
                # After stream ends, save the full bot response to DB
                if chatbot.short_term_memory:
                    ai_msg = Message(chatbot_id=chatbot_id, content=full_response, is_user=False)
                    db.add(ai_msg)
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

                # Save file with sanitized name
                file_id = str(uuid.uuid4())
                path = os.path.join(UPLOAD_DIR, f"{file_id}_{os.path.basename(file.filename)}")

                with open(path, "wb") as f:
                    f.write(file_content)

                db_doc = ChatbotDocument(
                    chatbot_id=chatbot_id,
                    filename=file.filename,
                    file_path=path,
                    content_type=file.content_type or "application/pdf"
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
                text = extract_pdf_text(doc.file_path)
                
                if not text or len(text.strip()) < 10:
                    logger.warning(f"Skipped unreadable PDF: {doc.file_path}")
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