# retrievers/hybrid_retriever.py

from .base import BaseRetriever

class HybridRetriever(BaseRetriever):
    def retrieve(self, query: str):
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        docs = retriever.invoke(query)

        if not docs:
            return [
                type("Doc", (), {"page_content": self.chatbot.system_prompt})()
            ]

        return docs