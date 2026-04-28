# rerankers/cross_encoder_reranker.py

from .base import BaseReranker
from sentence_transformers import CrossEncoder

class CrossEncoderReranker(BaseReranker):
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: list):
        pairs = [(query, getattr(doc, "page_content", "")) for doc in docs]

        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked]