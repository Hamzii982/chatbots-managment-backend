# rerankers/base.py

from abc import ABC, abstractmethod
from typing import List

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List) -> List:
        pass