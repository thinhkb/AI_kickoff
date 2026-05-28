"""Debug mismatched document questions."""
import sys, json, openpyxl
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load mismatches
mismatch_ids = set()
mismatches_data = {}
with open(PROJECT_ROOT / "outputs" / "eval_mismatches.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line.strip())
        if d["issue"] == "document_answer":
            mismatch_ids.add(d["id"])
            mismatches_data[d["id"]] = d

# Load question data
wb = openpyxl.load_workbook(PROJECT_ROOT / "data" / "example_data" / "example_data.xlsx")
ws = wb["example_question"]
headers = [c.value for c in ws[1]]

print("=" * 80)
print("MISMATCHED DOCUMENT QUESTIONS - DETAILED ANALYSIS")
print("=" * 80)

for row in ws.iter_rows(min_row=2, values_only=True):
    d = dict(zip(headers, row))
    rid = d["id"]
    if rid in mismatch_ids:
        mm = mismatches_data[rid]
        print(f"\n--- ID: {rid} ---")
        print(f"Question: {d['fun_question']}")
        print(f"Note: {d.get('note', 'NONE')}")
        print(f"Predicted: {mm['pred_answer']} | Ground Truth: {mm['gt_answer']}")

# Also check if questions mention specific doc IDs
print("\n" + "=" * 80)
print("DOC ID REFERENCES IN QUESTIONS")
print("=" * 80)
import re
for rid, mm in sorted(mismatches_data.items()):
    q = mm["question"]
    doc_match = re.search(r"Public[_\s-]*(\d+)", q, re.IGNORECASE)
    doc_id = f"Public_{int(doc_match.group(1)):03d}" if doc_match else "NO_DOC_ID"
    print(f"id={rid}: doc={doc_id} pred={mm['pred_answer']} gt={mm['gt_answer']}")

# Check chunk counts per mismatched doc
print("\n" + "=" * 80)
print("CHUNK COUNTS FOR MISMATCHED DOCS")
print("=" * 80)
chunks_file = PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"
doc_chunks = {}
with open(chunks_file, "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line.strip())
        did = c["doc_id"]
        if did not in doc_chunks:
            doc_chunks[did] = []
        doc_chunks[did].append(c)

for rid, mm in sorted(mismatches_data.items()):
    q = mm["question"]
    doc_match = re.search(r"Public[_\s-]*(\d+)", q, re.IGNORECASE)
    if doc_match:
        doc_id = f"Public_{int(doc_match.group(1)):03d}"
        chunks = doc_chunks.get(doc_id, [])
        avg_len = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
        print(f"id={rid} doc={doc_id}: {len(chunks)} chunks, avg_len={avg_len:.0f}")
    else:
        print(f"id={rid}: NO specific doc reference in question")
