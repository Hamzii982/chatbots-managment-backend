# rerankers/simple_reranker.py

from .base import BaseReranker

class SimpleReranker(BaseReranker):
    def rerank(self, query: str, docs: list):
        query_terms = set(query.lower().split())

        def score(doc):
            content = getattr(doc, "page_content", "").lower()
            return sum(1 for word in query_terms if word in content)

        return sorted(docs, key=score, reverse=True)