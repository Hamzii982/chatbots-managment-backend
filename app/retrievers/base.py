# retrievers/base.py

from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    def __init__(self, chatbot, vectorstore, db):
        self.chatbot = chatbot
        self.vectorstore = vectorstore
        self.db = db

    @abstractmethod
    def retrieve(self, query: str) -> List:
        pass