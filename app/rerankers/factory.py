# rerankers/factory.py

from .simple_reranker import SimpleReranker
from .cross_encoder_reranker import CrossEncoderReranker

def get_reranker(reranker_type: str):
    if reranker_type == "simple":
        return SimpleReranker()

    elif reranker_type == "cross":
        return CrossEncoderReranker()

    else:
        return None  # optional: no reranking