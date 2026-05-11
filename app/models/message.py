from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.db.base import Base

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    chatbot_id = Column(Integer, ForeignKey("chatbots.id"))
    thread_id = Column(Integer, ForeignKey("threads.id"), nullable=True)
    content = Column(String)
    is_user = Column(Boolean)  # True for user, False for AI
    created_at = Column(DateTime, server_default=func.now())