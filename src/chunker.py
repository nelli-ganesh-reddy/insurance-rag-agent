# src/chunker.py

import os
import json
import re
from pathlib import Path
from typing import List, Dict

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    EMBEDDING_MODEL, MAX_TOKENS, MIN_TOKENS,
    OVERLAP_RATIO, OUTPUT_DIR
)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
MODEL_MAX  = tokenizer.model_max_length


def token_count(text: str) -> int:
    return len(tokenizer(
        text, add_special_tokens=True,
        truncation=False, return_attention_mask=False
    )["input_ids"])


def split_by_tokens(text: str) -> List[str]:
    tokens = tokenizer(
        text, add_special_tokens=False,
        truncation=False, return_attention_mask=False
    )["input_ids"]
    chunks, step, start = [], int(MAX_TOKENS * (1 - OVERLAP_RATIO)), 0
    while start < len(tokens):
        chunk_tokens = tokens[start: start + MAX_TOKENS]
        if not chunk_tokens:
            break
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        start += step
    return chunks


def classify_clause(text: str) -> str:
    lower = text.lower()
    if "means" in lower and len(text.split()) < 120:
        return "definition"
    if any(w in lower for w in ["shall not", "not covered", "excludes",
                                 "we do not cover", "not payable",
                                 "not applicable", "not eligible"]):
        return "exclusion"
    if any(w in lower for w in ["subject to", "provided that", "condition"]):
        return "condition"
    if any(w in lower for w in ["limit", "maximum", "sum insured", "liability"]):
        return "limit"
    if any(w in lower for w in ["we will pay", "we cover", "indemnify"]):
        return "coverage"
    return "general"


def parse_markdown_tables(md_text: str) -> List[Dict]:
    tables  = []
    pattern = r"(\|.+?\|\n\|[-| ]+\|\n(?:\|.+?\|\n?)+)"
    for table in re.findall(pattern, md_text):
        lines = [l.strip() for l in table.strip().split("\n")]
        if len(lines) < 2:
            continue
        headers = [h.strip() for h in lines[0].split("|")[1:-1]]
        rows = []
        for row in lines[2:]:
            values = [v.strip() for v in row.split("|")[1:-1]]
            if len(values) != len(headers):
                continue
            rows.append("Row:\n" + "\n".join(
                f"- {h}: {v}" for h, v in zip(headers, values)
            ))
        tables.append({"headers": headers, "rows": rows})
    return tables


def process_pdf(input_path: str) -> str:
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {input_path}")

    print(f"[chunker] Converting: {input_path_obj.name} ...")

    converter = DocumentConverter()
    result    = converter.convert(str(input_path_obj))
    doc       = result.document

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    md_content        = doc.export_to_markdown()
    structured_tables = parse_markdown_tables(md_content)

    chunker     = HierarchicalChunker(merge_peers=False)
    base_chunks = list(chunker.chunk(doc))

    chunk_data, table_index, chunk_index = [], 0, 0

    for chunk in base_chunks:
        headings    = getattr(chunk.meta, "headings", [])
        doc_items   = getattr(chunk.meta, "doc_items", [])
        heading_ctx = " > ".join(headings) if headings else "General"
        chunk_text  = chunk.text.strip()

        is_table = any("table" in str(it.label).lower() for it in doc_items)

        if is_table and table_index < len(structured_tables):
            table_block = structured_tables[table_index]
            table_index += 1
            chunk_text = (
                f"Section: {heading_ctx}\n\n"
                f"This table contains policy information:\n\n"
                + "\n\n".join(table_block["rows"])
            )
        else:
            chunk_text = f"Section: {heading_ctx}\n\n{chunk_text}"

        sub_chunks = (
            split_by_tokens(chunk_text)
            if token_count(chunk_text) > MAX_TOKENS
            else [chunk_text]
        )

        for sub in sub_chunks:
            sub_tokens = token_count(sub)
            if sub_tokens < MIN_TOKENS:
                continue
            if sub_tokens >= MODEL_MAX:
                sub = tokenizer.decode(
                    tokenizer(sub, truncation=True,
                              max_length=MODEL_MAX - 5)["input_ids"]
                )
                sub_tokens = token_count(sub)

            page_number = None
            if hasattr(chunk.meta, "origin") and hasattr(chunk.meta.origin, "page_no"):
                page_number = chunk.meta.origin.page_no

            chunk_data.append({
                "index": chunk_index,
                "text":  sub,
                "metadata": {
                    "section_path": headings,
                    "doc_items":    [str(it.label) for it in doc_items],
                    "clause_type":  classify_clause(sub),
                    "token_count":  sub_tokens,
                    "page_number":  page_number
                }
            })
            chunk_index += 1

    out_path = Path(OUTPUT_DIR) / f"{input_path_obj.stem}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

    print(f"[chunker] {len(chunk_data)} chunks saved → {out_path}")
    return str(out_path)
