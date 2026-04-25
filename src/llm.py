# src/llm.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from huggingface_hub import InferenceClient
from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, HF_TOKEN, OUTPUT_DIR

FINETUNED_DIR = os.path.join(OUTPUT_DIR, "finetuned_model_phi2","checkpoint-676")
BASE_MODEL = "microsoft/phi-2"


def finetuned_model_exists() -> bool:
    return Path(FINETUNED_DIR).exists()

def get_llm_client() -> InferenceClient:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found. Add it to your .env file.")
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)


def ask(client, context: str, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an insurance assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, say "
                "'Not found in document'."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )

    return response.choices[0].message.content


def get_finetuned_client() -> dict:
    if not finetuned_model_exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {FINETUNED_DIR}.\n"
            "Run finetune.py first."
        )

    print(f"[llm] Loading fine-tuned Phi-2 from {FINETUNED_DIR} ...")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
    model.eval()

    print(f"[llm] Fine-tuned Phi-2 ready.")
    return {"model": model, "tokenizer": tokenizer}


def ask_finetuned(client: dict, context: str, query: str) -> str:
    import torch

    model = client["model"]
    tokenizer = client["tokenizer"]

    prompt = (
        f"Instruct: You are a helpful insurance assistant. "
        f"Answer clearly and concisely using ONLY the policy context below. "
        f"If the answer is not in the context, say 'Not found in document'.\n\n"
        f"Policy Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Output:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    bad_signals = [
        "<|endoftext|>", "<|user|>", "<|system|>",
        "Question:", "Context:", "Policy Context:",
        "Instruct:", "\n\nQ:", "Explanation:",
        "\n\nExplanation", "##", "**Rewrite",
        "Your task:", "Rewrite the above",
        "Output:"
    ]

    for signal in bad_signals:
        if signal in answer:
            answer = answer[:answer.index(signal)].strip()

    answer = answer.strip(" :,-\n\t")

    if not answer:
        return "Not found in document."

    return answer