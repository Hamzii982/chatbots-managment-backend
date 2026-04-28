from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from app.db.base import Base

class Chatbot(Base):
    __tablename__ = "chatbots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    system_prompt = Column(String)
    model_id = Column(Integer, ForeignKey("models.id"))
    
    retriever_type = Column(String, default="mock")
    chunk_size = Column(Integer, default=500)
    chunk_overlap = Column(Integer, default=50)
    reranker_type = Column(String, default="simple")
    top_k = Column(Integer, default=3)
    
    short_term_memory = Column(Boolean, default=True)
    long_term_memory = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    is_vectorized = Column(Boolean, default=False)