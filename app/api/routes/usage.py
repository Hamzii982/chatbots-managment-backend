from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.models.chatbot import Chatbot
from app.models.message import Message
from datetime import datetime, timedelta

router = APIRouter(prefix="/usage", tags=["usage"])

# Pricing per 1k tokens (Example: GPT-4o mini style pricing)
# In a real app, you might store these in the 'models' table
MODEL_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.002, "output": 0.006}
}

@router.get("/tokens")
def get_token_usage(db: Session = Depends(get_db)):
    # 1. Fetch messages with chatbot/model info joined
    # Note: This assumes your Message model has prompt_tokens and completion_tokens columns
    # If not, we use placeholders for the demo
    messages = db.query(
        Message, 
        Chatbot.name.label("bot_name"),
        Chatbot.model_id
    ).join(Chatbot, Message.chatbot_id == Chatbot.id).order_by(Message.created_at.desc()).limit(100).all()

    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    recent_logs = []
    model_counts = {}

    for msg, bot_name, model_id in messages:
        # Fallback values if columns don't exist yet
        p_tokens = getattr(msg, "prompt_tokens", 450) or 0
        c_tokens = getattr(msg, "completion_tokens", 150) or 0
        
        # Determine pricing (simplified logic)
        pricing = MODEL_PRICING.get("gpt-4o-mini") # Example fallback
        cost = ((p_tokens / 1000) * pricing["input"]) + ((c_tokens / 1000) * pricing["output"])

        total_prompt += p_tokens
        total_completion += c_tokens
        total_cost += cost

        # Track model distribution
        model_counts[bot_name] = model_counts.get(bot_name, 0) + (p_tokens + c_tokens)

        recent_logs.append({
            "time": msg.created_at,
            "bot_name": bot_name,
            "total_tokens": p_tokens + c_tokens,
            "cost": cost
        })

    # Calculate percentages for the breakdown chart
    grand_total_tokens = (total_prompt + total_completion) or 1
    model_breakdown = [
        {
            "model_name": name, 
            "percentage": round((count / grand_total_tokens) * 100, 1)
        } for name, count in model_counts.items()
    ]

    return {
        "total_cost": round(total_cost, 4),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "model_breakdown": model_breakdown[:3], # Top 3
        "recent_logs": recent_logs[:10] # Last 10
    }