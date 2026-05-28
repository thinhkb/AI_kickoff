import sys, json, re
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load chunks
chunks = []
with open(PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line.strip()))

# 1. Whitespace tokenization (current)
tokenized_ws = [c['text'].lower().split() for c in chunks]
bm25_ws = BM25Okapi(tokenized_ws)

# 2. Regex tokenization (proposed)
tokenized_regex = [re.findall(r'\w+', c['text'].lower(), flags=re.UNICODE) for c in chunks]
bm25_regex = BM25Okapi(tokenized_regex)

q = 'SASRec và BERT4Rec khác nhau chủ yếu ở những điểm nào?'

print("=== Whitespace Tokenization (Current) ===")
q_ws = q.lower().split()
scores_ws = bm25_ws.get_scores(q_ws)
top_ws = np.argsort(scores_ws)[::-1][:5]
for idx in top_ws:
    print(f"score={scores_ws[idx]:.4f} doc={chunks[idx]['doc_id']} page={chunks[idx]['page']}")
    print(f"  {chunks[idx]['text'][:100]}...")

print("\n=== Regex Tokenization (Proposed) ===")
q_regex = re.findall(r'\w+', q.lower(), flags=re.UNICODE)
scores_regex = bm25_regex.get_scores(q_regex)
top_regex = np.argsort(scores_regex)[::-1][:5]
for idx in top_regex:
    print(f"score={scores_regex[idx]:.4f} doc={chunks[idx]['doc_id']} page={chunks[idx]['page']}")
    print(f"  {chunks[idx]['text'][:100]}...")

