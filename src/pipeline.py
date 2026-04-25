# src/pipeline.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunker  import process_pdf
from src.embedder import (get_embedding_model, build_vectorstore,
                          load_vectorstore, vectorstore_exists)
from src.retriever import get_retriever, retrieve
from src.llm import (get_llm_client, ask,
                     get_finetuned_client, ask_finetuned,
                     finetuned_model_exists)


class InsuranceRAGPipeline:

    def __init__(self):
        self.embedding_model = None
        self.vectorstore     = None
        self.retriever       = None
        self.llm_client      = None
        self.use_finetuned   = False

    def build(self, pdf_path: str):
        print("\n=== Building pipeline ===")
        chunks_path          = process_pdf(pdf_path)
        self.embedding_model = get_embedding_model()
        self.vectorstore     = build_vectorstore(chunks_path, self.embedding_model)
        self.retriever       = get_retriever(self.vectorstore, k=8)
        self.llm_client      = get_llm_client()
        self.use_finetuned   = False
        print("=== Pipeline ready (Llama 3 API) ===\n")

    def load(self, use_finetuned: bool = False):

        if not vectorstore_exists():
            raise FileNotFoundError(
                "FAISS store not found. Run pipeline.build('data/your_policy.pdf') first."
            )

        print("\n=== Loading pipeline ===")
        self.embedding_model = get_embedding_model()
        self.vectorstore     = load_vectorstore(self.embedding_model)

        if use_finetuned:
            if not finetuned_model_exists():
                raise FileNotFoundError(
                    "Fine-tuned model not found. Run finetune.py first."
                )
            self.retriever     = get_retriever(self.vectorstore, k=3)  
            self.llm_client    = get_finetuned_client()
            self.use_finetuned = True
            print("=== Pipeline ready (fine-tuned Phi-2, k=3 chunks) ===\n")
        else:
            self.retriever     = get_retriever(self.vectorstore, k=8)
            self.llm_client    = get_llm_client()
            self.use_finetuned = False
            print("=== Pipeline ready (Llama 3 API, k=8 chunks) ===\n")

    def ask(self, query: str) -> str:
        if not self.retriever:
            raise RuntimeError("Pipeline not loaded. Call build() or load() first.")

        docs, context = retrieve(self.retriever, query)

        if self.use_finetuned:
            return ask_finetuned(self.llm_client, context, query)
        else:
            return ask(self.llm_client, context, query)
