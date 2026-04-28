# retrievers/mock_retriever.py

from .base import BaseRetriever

class MockRetriever(BaseRetriever):
    def retrieve(self, query: str):
        return [
            type("Doc", (), {"page_content": "This is a mock response document."})()
        ]