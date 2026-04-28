from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.models.chatbot import Chatbot
from app.models.model_config import ModelConfig
from app.models.message import Message

router = APIRouter(prefix="/statistics", tags=["statistics"])

@router.get("/summary")
def get_summary_stats(db: Session = Depends(get_db)):
    # 1. Total Message Count
    total_messages = db.query(Message).count()
    
    total_models = db.query(ModelConfig).count()

    # 2. Count Active Chatbots
    active_bots = db.query(Chatbot).filter(Chatbot.is_active == True).count()

    # 3. Aggregate Messages per Chatbot
    bot_metrics_raw = db.query(
        Chatbot.name, 
        func.count(Message.id).label("count")
    ).join(Message, Message.chatbot_id == Chatbot.id)\
     .group_by(Chatbot.name)\
     .order_by(func.count(Message.id).desc()).all()

    bot_metrics = [{"name": m[0], "count": m[1]} for m in bot_metrics_raw]

    return {
        "total_messages": total_messages,
        "total_models": total_models,
        "active_bots": active_bots,
        "avg_latency": 0.45,  # Placeholder until you implement actual timing logic
        "error_count": 0,     # Placeholder for error logging
        "bot_metrics": bot_metrics
    }