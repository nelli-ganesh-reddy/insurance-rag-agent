# config.py

import os
import torch
from dotenv import load_dotenv

load_dotenv()

DATA_DIR   = "./data"
OUTPUT_DIR = "./output"
FAISS_DIR  = "./output/policy_faiss"

EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

MAX_TOKENS    = 160
MIN_TOKENS    = 30       
OVERLAP_RATIO = 0.25

RETRIEVER_K       = 8    
RETRIEVER_FETCH_K = 20

LLM_MODEL       = "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_MAX_TOKENS  = 300
LLM_TEMPERATURE = 0.2
HF_TOKEN        = os.getenv("HF_TOKEN")
