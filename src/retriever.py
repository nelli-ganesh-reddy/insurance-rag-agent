# src/retriever.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import FAISS
from config import RETRIEVER_K, RETRIEVER_FETCH_K


def get_retriever(vectorstore: FAISS,
                  k: int = RETRIEVER_K,
                  fetch_k: int = RETRIEVER_FETCH_K):

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k}
    )


def retrieve(retriever, query: str) -> tuple:
    docs    = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return docs, context
