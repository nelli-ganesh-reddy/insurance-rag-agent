# src/finetune.py

import os
import sys
import random
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from config import OUTPUT_DIR


BASE_MODEL = "microsoft/phi-2"
FINETUNED_DIR = os.path.join(OUTPUT_DIR, "finetuned_model_phi2")

HF_DATASET = "deccan-ai/insuranceQA-v2"
NUM_SAMPLES = 3000

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 5e-4
MAX_SEQ_LEN = 768

def format_sample(sample: dict) -> str:
    question = sample.get("input", "").strip()
    answer = sample.get("output", "").strip()

    if not question or not answer:
        return ""

    return (
        f"Instruct: You are a helpful insurance assistant. "
        f"Answer clearly and concisely.\n\n"
        f"Question: {question}\n\n"
        f"Output: {answer}<|endoftext|>"
    )

def is_clean_sample(sample: dict) -> bool:
    answer = sample.get("output", "").strip()

    if len(answer) < 50:
        return False
    if len(answer) > 600:
        return False

    answer_lower = answer.lower()

    noise_signals = [
        "@", "www.", "http", "phone", "call us", "contact us",
        "contact me", "please call", "email", "website",
        "800-", "866-", "877-", "888-", "727-", "1-800",
        "thank you very much", "thanks very much",
        "thanks!", "thank you!", "thanks again", "once again thank",
        "feel free to contact", "please feel free",
        "good luck", "best wishes", "hope this helps",
        "have a great", "take care", "bye bye", "cheers!",
        "look forward", "we look forward",
        "hope to hear", "hear back soon",
        "president", "ceo", "director", "broker", "agent",
        "signed by", "registered office", "vat no",
        "insurance brokerage", "brokerage services",
        "gary", "krishna", "seltzer", "selby", "robyn", "niki",
        "my name is", "i would be happy", "i am happy to",
        "-lrb-", "-rrb-", "lrb-", "rrb-",
        ":d", ":)", ":lrb", "lol ", "xxx", " xo",
        "thanks for asking", "great question",
        "feel free to ask", "don't hesitate",
        "##your task", "**rewrite", "rewrite the above",
        "elementary school", "summarization",
        "your task:", "rewrite the above paragraph",
    ]

    for signal in noise_signals:
        if signal in answer_lower:
            return False

    last_line = answer.strip().split("\n")[-1].strip()
    words = [w for w in last_line.split() if w]
    if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if len(w) > 0):
        return False

    return True


def load_hf_dataset() -> Dataset:
    print(f"[finetune] Loading dataset: {HF_DATASET} ...")
    print(f"[finetune] This downloads ~50MB from HuggingFace....wait a moment...\n")

    ds = load_dataset(HF_DATASET, split="train")
    print(f"[finetune] Full dataset size: {len(ds)} samples")

    print(f"[finetune] Filtering noisy samples ...")
    clean_ds = [s for s in ds if is_clean_sample(s)]
    print(f"[finetune] Clean samples: {len(clean_ds)} / {len(ds)}")

    random.seed(42)
    random.shuffle(clean_ds)

    clean_ds = clean_ds[:min(NUM_SAMPLES, len(clean_ds))]
    print(f"[finetune] Using: {len(clean_ds)} samples for training\n")

    formatted = []
    skipped = 0

    for sample in clean_ds:
        text = format_sample(sample)
        if text:
            formatted.append({"text": text})
        else:
            skipped += 1

    print(f"[finetune] Formatted: {len(formatted)} samples")
    print(f"[finetune] Skipped (empty): {skipped} samples")

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"[finetune] Train: {len(split['train'])} | Eval: {len(split['test'])}")
    return split


def load_base_model():
    print(f"[finetune] Loading base model: {BASE_MODEL} ...")
    print(f"[finetune] Downloads ~5GB from HuggingFace — takes 2-3 minutes...\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print(f"[finetune] Base model loaded.")
    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"[finetune] Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model

def tokenize_dataset(dataset_split, tokenizer):
    pad_id = tokenizer.pad_token_id

    def tokenize(sample):
        full_text = sample["text"]
        split_marker = "Output:"

        if split_marker not in full_text:
            result = tokenizer(
                full_text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length"
            )
            labels = [-100 if tok == pad_id else tok for tok in result["input_ids"]]
            result["labels"] = labels
            return result

        prompt_part = full_text[:full_text.index(split_marker) + len(split_marker)]
        prompt_tokens = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        result = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )

        input_ids = result["input_ids"]
        labels = []

        for i, tok in enumerate(input_ids):
            if i < prompt_len:
                labels.append(-100)
            elif tok == pad_id:
                labels.append(-100)
            else:
                labels.append(tok)

        result["labels"] = labels
        return result

    return dataset_split.map(tokenize, batched=False, remove_columns=["text"])


def train(model, tokenizer, dataset_split):
    print(f"\n[finetune] Starting training ...")
    print(f"[finetune] Epochs:{NUM_EPOCHS} | LR:{LEARNING_RATE} | LoRA_R:{LORA_R}\n")

    os.makedirs(FINETUNED_DIR, exist_ok=True)
    tokenized = tokenize_dataset(dataset_split, tokenizer)

    args = TrainingArguments(
        output_dir=FINETUNED_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_pin_memory=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()
    return trainer

def save_model(trainer, tokenizer):
    print(f"\n[finetune] Saving model to {FINETUNED_DIR} ...")
    trainer.model.save_pretrained(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    print(f"[finetune] Saved! → {FINETUNED_DIR}")

def run_finetuning():
    print(f"\n[finetune] === Fine-tuning Phi-2 on {HF_DATASET} ===")
    print(f"[finetune] Samples: {NUM_SAMPLES} | Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE}\n")

    dataset_split = load_hf_dataset()
    model, tokenizer = load_base_model()
    model = apply_lora(model)
    trainer = train(model, tokenizer, dataset_split)
    save_model(trainer, tokenizer)

    print(f"\n[finetune] Done!")
    print(f"[finetune] Model saved to: {FINETUNED_DIR}")
    print(f"[finetune] Now run: pipeline.load(use_finetuned=True)")