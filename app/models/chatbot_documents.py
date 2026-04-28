from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from datetime import datetime
from app.db.base import Base

class ChatbotDocument(Base):
    __tablename__ = "chatbot_documents"

    id = Column(Integer, primary_key=True, index=True)
    chatbot_id = Column(Integer, ForeignKey("chatbots.id"))

    filename = Column(String)
    file_path = Column(String)   # stored path on disk or S3
    content_type = Column(String)

    uploaded_at = Column(DateTime, default=datetime.utcnow)