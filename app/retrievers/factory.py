# retrievers/factory.py

from .vector_retriever import VectorRetriever
from .mock_retriever import MockRetriever
from .hybrid_retriever import HybridRetriever

def get_retriever(chatbot, vectorstore, db):
    retriever_type = chatbot.retriever_type

    if retriever_type == "vector":
        return VectorRetriever(chatbot, vectorstore, db)

    elif retriever_type == "mock":
        return MockRetriever(chatbot, vectorstore, db)

    elif retriever_type == "hybrid":
        return HybridRetriever(chatbot, vectorstore, db)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")