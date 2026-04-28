# retrievers/vector_retriever.py

from .base import BaseRetriever

class VectorRetriever(BaseRetriever):
    def retrieve(self, query: str):
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        return retriever.invoke(query)