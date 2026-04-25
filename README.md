# InsureRAG v2 — Insurance Policy Q&A using RAG + Fine-tuning

An AI-powered question answering system for insurance policy documents. It uses **Retrieval-Augmented Generation (RAG)** with **Llama 3** and a **fine-tuned Phi-2** model trained on real insurance QA data for comparison.

---

## What it does

- Upload any insurance policy PDF
- Ask questions in plain English
- Get accurate answers pulled directly from the policy document
- Compare answers from two models: **Llama 3 (RAG)** vs **Fine-tuned Phi-2**
- Interact via a clean web UI (`index.html`) served through a Flask API + ngrok tunnel

---

## How it works

### RAG Pipeline

```
PDF → Chunker → Embeddings (BAAI/bge) → FAISS Vector Store
                                                ↓
                         User Question → Retriever (MMR) → Top-K Chunks
                                                                  ↓
                                                  Context + Question → Llama 3 API → Answer
```

### Fine-tuning Pipeline

```
deccan-ai/insuranceQA-v2 (HuggingFace dataset)
        ↓
  Filter noisy samples (~8k–12k clean out of 27k)
        ↓
  Take 3000 samples → Format as Instruct/Output pairs
        ↓
  Load microsoft/phi-2 in 4-bit (QLoRA)
        ↓
  Apply LoRA adapters (r=8, alpha=16)
        ↓
  Train 1 epoch on Colab T4 GPU (~1–2 hours)
        ↓
  Save fine-tuned model → output/finetuned_model_phi2
```

---

## Project Structure

```
insurance_rag_v2/
├── src/
│   ├── chunker.py                  # PDF → smart text chunks using Docling
│   ├── embedder.py                 # Chunks → FAISS vector store (BAAI/bge embeddings)
│   ├── retriever.py                # MMR-based semantic retrieval
│   ├── llm.py                      # Llama 3 API inference + fine-tuned Phi-2 inference
│   ├── finetune.py                 # QLoRA fine-tuning of Phi-2 on insuranceQA-v2
│   └── pipeline.py                 # End-to-end pipeline (build / load / ask)
├── data/
│   └── policy-wordings.pdf         # Sample insurance policy PDF
├── output/                         # Generated at runtime (FAISS index, chunks, model)
├── config.py                       # All settings in one place
├── requirements.txt
├── AI_PROJECT.ipynb                # Main Colab notebook
├── colab_notebook_instructions.py  # Step-by-step Colab cell guide
├── index.html                      # Web UI
└── .env                            # Your HuggingFace token (never push this)
```

---

## Models Used

| Model | Purpose |
|-------|---------|
| `BAAI/bge-base-en-v1.5` | Embedding chunks into vector space |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Answering questions via HuggingFace Inference API |
| `microsoft/phi-2` | Base model that gets fine-tuned on insurance QA data |

---

## Fine-tuning Details (`src/finetune.py`)

The fine-tuning uses **QLoRA** (Quantized Low-Rank Adaptation) to efficiently train Phi-2 on a GPU with limited memory.

| Setting | Value |
|---------|-------|
| Dataset | `deccan-ai/insuranceQA-v2` (~27k samples) |
| Clean samples used | ~3000 (after noise filtering) |
| Base model | `microsoft/phi-2` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Epochs | 1 |
| Learning rate | 5e-4 |
| Batch size | 2 (grad accum = 4) |
| Max sequence length | 768 tokens |
| Optimizer | paged_adamw_8bit |
| GPU required | T4 (Colab free tier works) |
| Training time | ~1–2 hours |

**Noise filtering** — the dataset has a lot of real-world messy answers (phone numbers, emails, agent names, filler phrases like "thanks!", etc.). `finetune.py` filters these out before training to keep only clean, factual insurance answers.

**Training format:**
```
Instruct: You are a helpful insurance assistant. Answer clearly and concisely.

Question: What is the grace period for premium payment?

Output: The grace period is 30 days after the due date for payment of premium...<|endoftext|>
```

Only the **Output** part is used for loss calculation — the prompt is masked with `-100` labels so the model learns to generate answers, not repeat questions.

---

## Running the Project

> ⚠️ **This project is designed to run on Google Colab** because:
> - Fine-tuning Phi-2 requires a GPU (Colab provides a free T4)
> - The web UI connects to the model via an ngrok tunnel from Colab
> - `bitsandbytes` 4-bit quantization needs CUDA

### Step 1 — Upload to Google Drive

Upload the entire `insurance_rag_v2/` folder to your Google Drive root:
```
MyDrive/insurance_rag_v2/
```

### Step 2 — Open Colab

Open `AI_PROJECT.ipynb` in Google Colab and enable GPU:
**Runtime → Change runtime type → T4 GPU**

### Step 3 — Run cells in order

| Cell | What it does | When to run |
|------|-------------|-------------|
| Cell 1 | Mount Google Drive + go to project folder | Every session |
| Cell 2 | Install all packages | Every session |
| Cell 3 | Set HuggingFace token | Every session |
| Cell 4 | Build RAG pipeline from PDF | Once only |
| Cell 5 | Load saved RAG pipeline | Every session after Cell 4 |
| Cell 6 | Test RAG with Llama 3 | Anytime |
| Cell 7 | Fine-tune Phi-2 on insuranceQA-v2 (~1–2 hrs) | Once only |
| Cell 8 | Load fine-tuned Phi-2 pipeline | After Cell 7 |
| Cell 9 | Demo: Llama 3 RAG vs Fine-tuned Phi-2 | Anytime |
| Cell 14–15 | Start Flask API + ngrok for Web UI | After pipelines loaded |

### Step 4 — Use the Web UI

After running the Flask + ngrok cell, copy the ngrok URL printed in the output and update `index.html`:

```javascript
const API_URL = "https://your-ngrok-url.ngrok-free.app";
```

Then open `index.html` in your browser.

---

## Setup

### HuggingFace Token

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

In Colab, set it in Cell 3:
```python
os.environ["HF_TOKEN"] = "hf_your_token_here"
```

For local use, create a `.env` file:
```
HF_TOKEN=hf_your_token_here
```

### Configuration (`config.py`)

```python
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
LLM_MODEL        = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_TOKENS       = 160       # chunk size
RETRIEVER_K      = 8         # chunks retrieved per query
LLM_MAX_TOKENS   = 300       # max answer length
LLM_TEMPERATURE  = 0.2
```

---

## Requirements

```
docling, transformers, langchain, langchain-community
langchain-huggingface, sentence-transformers, faiss-cpu
huggingface-hub, python-dotenv, datasets, peft, trl
bitsandbytes, accelerate
```

```bash
pip install -r requirements.txt
```

---

## Notes

- The `output/` folder stores FAISS indexes, chunk JSONs, and fine-tuned model checkpoints — all saved to Google Drive so you don't rebuild every session
- Fine-tuning saves to `output/finetuned_model_phi2/` — this can be **several GBs**, avoid pushing it to GitHub
- The `.env` file is gitignored — never push your HuggingFace token
