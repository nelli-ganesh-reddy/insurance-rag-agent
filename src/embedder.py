# src/embedder.py

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, FAISS_DIR


def get_embedding_model() -> HuggingFaceEmbeddings:
    print(f"[embedder] Loading embedding model: {EMBEDDING_MODEL} ...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_chunks(chunks_json_path: str) -> list:
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents   = []
    source_name = Path(chunks_json_path).stem.replace("_chunks", "")

    for chunk in chunks:
        if not chunk.get("text"):
            continue
        metadata = chunk.get("metadata", {})
        metadata["source"]  = source_name
        metadata["section"] = " > ".join(metadata.get("section_path", []))
        documents.append(Document(page_content=chunk["text"], metadata=metadata))

    print(f"[embedder] Loaded {len(documents)} documents.")
    return documents


def build_vectorstore(chunks_json_path: str,
                      embedding_model: HuggingFaceEmbeddings) -> FAISS:
    documents   = load_chunks(chunks_json_path)
    print(f"[embedder] Building FAISS vector store ...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(FAISS_DIR)
    print(f"[embedder] Vector store saved → {FAISS_DIR}")
    return vectorstore


def load_vectorstore(embedding_model: HuggingFaceEmbeddings) -> FAISS:
    print(f"[embedder] Loading FAISS store from {FAISS_DIR} ...")
    return FAISS.load_local(
        FAISS_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )


def vectorstore_exists() -> bool:
    return Path(FAISS_DIR).exists()
