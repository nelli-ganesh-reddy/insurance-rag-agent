# colab_notebook.py
# ================================================================
# COMPLETE COLAB NOTEBOOK — insurance_rag_v2
# Copy each block into a separate Colab cell and run in order.
# ================================================================
#
# CELL 1  → Run EVERY session  — Mount Drive + go to folder
# CELL 2  → Run EVERY session  — Install packages
# CELL 3  → Run EVERY session  — Set HF token
# CELL 4  → Run ONCE           — Build RAG pipeline from your PDF
# CELL 5  → Run every session  — Load RAG pipeline (after Cell 4 done)
# CELL 6  → Test RAG with Llama 3
# CELL 7  → Run ONCE           — Fine-tune Phi-2 (1-2 hours)
# CELL 8  → After Cell 7 done  — Load fine-tuned pipeline
# CELL 9  → Demo comparison
# ================================================================


# ────────────────────────────────────────────────────────────────
# CELL 1 — Mount Drive + go to project folder
# Run this EVERY Colab session — Colab resets on restart
# ────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/insurance_rag_v2')
print("Working directory:", os.getcwd())
# Expected: /content/drive/MyDrive/insurance_rag_v2


# ────────────────────────────────────────────────────────────────
# CELL 2 — Install all packages
# Run this EVERY Colab session
#
# FIX #12: Added faiss-gpu (matches EMBEDDING_DEVICE="cuda" in config.py).
#          If you're on a CPU-only runtime, replace faiss-gpu with faiss-cpu.
# ────────────────────────────────────────────────────────────────
# !pip install docling transformers langchain langchain-community \
#              langchain-huggingface sentence-transformers faiss-gpu \
#              huggingface-hub python-dotenv datasets peft trl \
#              bitsandbytes accelerate -q


# ────────────────────────────────────────────────────────────────
# CELL 3 — Set HuggingFace token
# Run this EVERY Colab session
# Get token: huggingface.co → Settings → Access Tokens
# ────────────────────────────────────────────────────────────────
import os
os.environ["HF_TOKEN"] = "hf_paste_your_token_here"


# ────────────────────────────────────────────────────────────────
# CELL 4 — Build RAG pipeline from your PDF (ONCE only)
# Upload your PDF to data/ folder in Drive first.
# Change "your_policy.pdf" to your actual PDF filename.
# Takes 10–15 minutes.
# ────────────────────────────────────────────────────────────────
from src.pipeline import InsuranceRAGPipeline

pipeline = InsuranceRAGPipeline()
pipeline.build("data/your_policy.pdf")   # ← change this filename

# Expected output:
# [chunker] Converting: your_policy.pdf ...
# [chunker] X chunks saved → output/your_policy_chunks.json
# [embedder] Building FAISS vector store ...
# [embedder] Vector store saved → output/policy_faiss
# === Pipeline ready (Llama 3 API) ===


# ────────────────────────────────────────────────────────────────
# CELL 5 — Load RAG pipeline (every session AFTER Cell 4)
# Use this instead of Cell 4 once you have built the pipeline once.
# Takes ~30 seconds.
# ────────────────────────────────────────────────────────────────
from src.pipeline import InsuranceRAGPipeline

pipeline = InsuranceRAGPipeline()
pipeline.load()   # loads saved FAISS from Drive


# ────────────────────────────────────────────────────────────────
# CELL 6 — Test RAG pipeline with Llama 3
# Run after Cell 4 or Cell 5
# ────────────────────────────────────────────────────────────────
questions = [
    "What is the definition of critical illness?",
    "What is the grace period for premium payment?",
    "Does the policy cover HIV/AIDS?",
    "What treatments are covered under dental expenses?",
    "What are the conditions for discharge from hospital?",
]

print("=== RAG Pipeline — Llama 3 Answers ===\n")
for q in questions:
    print(f"Q: {q}")
    print(f"A: {pipeline.ask(q)}\n")
    print("-" * 60)


# ────────────────────────────────────────────────────────────────
# CELL 7 — Fine-tune Phi-2 on deccan-ai/insuranceQA-v2 (ONCE only)
# IMPORTANT: Make sure T4 GPU is enabled before running this.
# Runtime → Change runtime type → T4 GPU
# Takes 1–2 hours.
# ────────────────────────────────────────────────────────────────
from src.finetune import run_finetuning

run_finetuning()

# Expected output:
# [finetune] Loading dataset: deccan-ai/insuranceQA-v2 ...
# [finetune] Full dataset size: 27987 samples
# [finetune] Clean samples: ~8000-12000 / 27987  ← more filtered now
# [finetune] Using: 3000 samples for training
# [finetune] Loading base model: microsoft/phi-2 ...
# [finetune] Trainable: X / Y (0.26%)
# [finetune] Starting training ...
# Epoch 1: loss = X.XX
# Epoch 2: loss = X.XX
# [finetune] Saved! → output/finetuned_model_phi2


# ────────────────────────────────────────────────────────────────
# CELL 8 — Load fine-tuned Phi-2 pipeline (after Cell 7 done)
# Now uses k=3 chunks instead of k=2 for better context coverage.
# ────────────────────────────────────────────────────────────────
from src.pipeline import InsuranceRAGPipeline

pipeline_ft = InsuranceRAGPipeline()
pipeline_ft.load(use_finetuned=True)   # loads fine-tuned Phi-2

# Expected:
# [llm] Loading fine-tuned Phi-2 from output/finetuned_model_phi2 ...
# [llm] Fine-tuned Phi-2 ready.
# === Pipeline ready (fine-tuned Phi-2, k=3 chunks) ===


# ────────────────────────────────────────────────────────────────
# CELL 9 — DEMO: Llama 3 RAG vs Fine-tuned Phi-2 comparison
# This is what you show your professor / evaluator
# ────────────────────────────────────────────────────────────────
from src.pipeline import InsuranceRAGPipeline

# Load both pipelines
pipeline_rag = InsuranceRAGPipeline()
pipeline_rag.load()                      # Llama 3

pipeline_ft = InsuranceRAGPipeline()
pipeline_ft.load(use_finetuned=True)     # Fine-tuned Phi-2

# Demo questions
demo_questions = [
    "What is the definition of critical illness?",
    "What is the grace period for premium payment?",
    "Does the policy cover HIV/AIDS?",
    "What treatments are covered under dental expenses?",
    "What are the conditions for discharge from hospital?",
]

print("=" * 65)
print("DEMO: RAG (Llama 3 API)  vs  Fine-tuned Phi-2")
print("=" * 65)

for q in demo_questions:
    print(f"\nQuestion: {q}\n")
    print(f"[Llama 3 via RAG]:\n{pipeline_rag.ask(q)}\n")
    print(f"[Fine-tuned Phi-2]:\n{pipeline_ft.ask(q)}")
    print("-" * 65)
