import json

with open(r'g:\AI_kickoff\outputs\predictions_v4.jsonl', 'r', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f if line.strip()]

api_rows = [r for r in rows if r['function_code'] == 'call_api']
doc_rows = [r for r in rows if r['function_code'] == 'call_document']

api_times = [r['time_response'] for r in api_rows]
doc_times = [r['time_response'] for r in doc_rows]

print(f"=== call_api ({len(api_rows)} rows) ===")
print(f"  min:  {min(api_times):.4f}s")
print(f"  max:  {max(api_times):.4f}s")
print(f"  mean: {sum(api_times)/len(api_times):.4f}s")
print(f"  zero: {sum(1 for t in api_times if t == 0)} / {len(api_times)}")

print(f"\n=== call_document ({len(doc_rows)} rows) ===")
print(f"  min:  {min(doc_times):.4f}s")
print(f"  max:  {max(doc_times):.4f}s")
print(f"  mean: {sum(doc_times)/len(doc_times):.4f}s")
print(f"  zero: {sum(1 for t in doc_times if t == 0)} / {len(doc_times)}")

# Show first 10 API times
print(f"\n=== API time samples ===")
for r in api_rows[:10]:
    print(f"  id={r['id']}, time={r['time_response']:.4f}s")
