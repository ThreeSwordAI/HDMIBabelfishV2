#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JP→EN translation evaluation with fixed issues + CSV logging.

New: saves results into results.csv by default.
"""

import argparse
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

# ----------------------------
# Model registry
# ----------------------------
MODELS: Dict[str, Dict] = {
    "marian_opus_ja_en": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "type": "marian",
    },
    "marian_opus_ja_en_finetuned": {
        "model_name": "Helsinki-NLP/opus-mt-ja-en",
        "local_path": "./models/marian_opus_ja_en_finetuned",
        "type": "marian",
    },
    "marian_jap_en": {
        "model_name": "Helsinki-NLP/opus-mt-jap-en",
        "type": "marian",
    },
    "nllb_600m": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "type": "nllb",
        "src_lang": "jpn_Jpan",
        "tgt_lang": "eng_Latn",
    },
}

# ----------------------------
# Device
# ----------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
torch.manual_seed(42)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(42)

# ----------------------------
# Data loading
# ----------------------------
JP_CHAR_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3000-\u303f]")
ASCII_RE = re.compile(r"^[\x00-\x7F]+$")

def is_mostly_japanese(text: str, threshold: float = 0.05) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    jp = len(JP_CHAR_RE.findall(text))
    return (jp / max(len(text), 1)) >= threshold

def detect_jp_en_columns(df: pd.DataFrame) -> Tuple[str, str]:
    text_cols = [c for c in df.columns if df[c].dtype == object]
    jp_scores = {c: df[c].astype(str).apply(is_mostly_japanese).mean() for c in text_cols}
    jp_col = max(jp_scores, key=jp_scores.get)
    ascii_scores = {c: df[c].astype(str).apply(lambda s: bool(ASCII_RE.match(s.strip()))).mean()
                    for c in text_cols if c != jp_col}
    en_col = max(ascii_scores, key=ascii_scores.get)
    return jp_col, en_col

def load_pairs(csv_path: str, limit: int = 0) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    candidates = [c.lower() for c in df.columns]
    if {"jp", "eng"}.issubset(set(candidates)):
        jp_col = df.columns[candidates.index("jp")]
        en_col = df.columns[candidates.index("eng")]
    else:
        jp_col, en_col = detect_jp_en_columns(df)
    src = df[jp_col].astype(str).fillna("").tolist()
    ref = df[en_col].astype(str).fillna("").tolist()
    if limit > 0:
        src, ref = src[:limit], ref[:limit]
    return src, ref

# ----------------------------
# Normalization
# ----------------------------
def normalize_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").strip()
    return s

# ----------------------------
# Model loading
# ----------------------------
def resolve_repo_or_local(model_cfg: Dict) -> Tuple[str, bool]:
    local_path = model_cfg.get("local_path")
    if local_path and Path(local_path).exists():
        return local_path, True
    return model_cfg["model_name"], False

@torch.inference_mode()
def load_model(model_key: str):
    cfg = MODELS[model_key]
    repo_or_local, local_only = resolve_repo_or_local(cfg)
    tokenizer = AutoTokenizer.from_pretrained(repo_or_local, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_or_local, local_files_only=local_only)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, cfg

# ----------------------------
# Translation
# ----------------------------
GEN_KW = dict(num_beams=4, max_new_tokens=120, early_stopping=True)

def translate_batch(texts: List[str], tokenizer, model, cfg: Dict, batch_size: int = 8) -> List[str]:
    outs: List[str] = []
    forced_bos_token_id = None
    if cfg["type"] == "nllb":
        tokenizer.src_lang = cfg["src_lang"]
        forced_bos_token_id = tokenizer.lang_code_to_id[cfg["tgt_lang"]]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        if cfg["type"] == "mt5":
            chunk = [f"translate Japanese to English: {t}" for t in chunk]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        gen_kwargs = dict(GEN_KW)
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
        output_ids = model.generate(**inputs, **gen_kwargs)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outs.extend([normalize_en(x) for x in decoded])
        del inputs, output_ids
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return outs

# ----------------------------
# Metrics
# ----------------------------
def compute_bleu(hyp: List[str], ref: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hyp, [ref], tokenize="13a")
    return float(bleu.score)

def model_param_size_mb(model: torch.nn.Module) -> float:
    dtype_size = 2 if next(model.parameters()).dtype in (torch.float16, torch.bfloat16) else 4
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * dtype_size) / (1024**2)

# ----------------------------
# Runner
# ----------------------------
def evaluate_model(model_key: str, src: List[str], ref: List[str], n_samples: int) -> Dict:
    print(f"\nEvaluating {model_key}...")
    tokenizer, model, cfg = load_model(model_key)
    start = time.time()
    hyp = translate_batch(src, tokenizer, model, cfg, batch_size=8)
    elapsed = time.time() - start
    bleu = compute_bleu(hyp, ref)
    size_mb = model_param_size_mb(model)
    del tokenizer, model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": model_key,
        "bleu": round(bleu, 2),
        "time_sec": round(elapsed, 4),
        "size_mb": round(size_mb, 1),
        "samples": n_samples,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--which", type=str, default="all")
    parser.add_argument("--out", type=str, default="results.csv")
    args = parser.parse_args()

    src, ref = load_pairs(args.csv, limit=args.limit)
    n = len(src)
    print(f"Testing on {n} samples...")

    keys = list(MODELS.keys()) if args.which.strip().lower() == "all" else [k.strip() for k in args.which.split(",")]

    all_results = []
    for idx, key in enumerate(keys, 1):
        print(f"\n[{idx}/{len(keys)}] Processing {key}...")
        res = evaluate_model(key, src, ref, n)
        print(f"✅ {res['model']}: BLEU={res['bleu']}, Time={res['time_sec']}s, Size={res['size_mb']}MB")
        all_results.append(res)

    # Save to CSV
    out_path = Path(args.out)
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\n📂 Results saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
