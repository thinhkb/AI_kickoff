# 🤖 Viettel AI Race – Smart Chatbot for RAG & API Integration

## Mục lục

- [1. Tổng quan](#1-tổng-quan)
- [2. Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
- [3. Cài đặt](#3-cài-đặt)
- [4. Cấu trúc dữ liệu](#4-cấu-trúc-dữ-liệu)
- [5. Hướng dẫn chạy project](#5-hướng-dẫn-chạy-project)
- [6. Kiến trúc hệ thống](#6-kiến-trúc-hệ-thống)
- [7. Cấu trúc project](#7-cấu-trúc-project)
- [8. Chi tiết từng module](#8-chi-tiết-từng-module)
- [9. Cấu hình](#9-cấu-hình)
- [10. Kết quả đạt được](#10-kết-quả-đạt-được)
- [11. Nâng cấp nâng cao](#11-nâng-cấp-nâng-cao)
- [12. Xử lý lỗi thường gặp](#12-xử-lý-lỗi-thường-gặp)

---

## 1. Tổng quan

Hệ thống chatbot thông minh **hai nhánh** cho cuộc thi Viettel AI Race:

| Nhánh | Chức năng | Input | Output |
|-------|-----------|-------|--------|
| `call_document` | Trả lời câu hỏi trắc nghiệm từ tài liệu PDF | `question` + `note` (A/B/C/D) | `{"numbers": 1, "result": "A"}` |
| `call_api` | Chọn API đúng và điền JSON config | `question` | `{"path": "/api/...", "body": {...}}` |

**Nguyên tắc thiết kế:**
- Chỉ dùng `question` để phân loại nhánh (không dùng `note`)
- Không fallback giữa hai nhánh
- Ưu tiên tốc độ: retrieval + scoring + template filling (hạn chế LLM)

---

## 2. Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|------------|-------------------|
| **Python** | >= 3.10 |
| **RAM** | >= 8 GB |
| **Disk** | >= 2 GB (cho data + models) |
| **GPU** | Không bắt buộc (chỉ cần cho embedding/reranker nâng cao) |
| **OS** | Windows / Linux / macOS |

---

## 3. Cài đặt

### 3.1. Clone project

```bash
git clone <repo-url>
cd AI_kickoff
```

### 3.2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3.3. Cài đặt dependencies

**Cài đặt cơ bản** (đủ để chạy pipeline):

```bash
pip install openpyxl scikit-learn numpy scipy rapidfuzz rank-bm25 tqdm PyMuPDF joblib
```

**Cài đặt đầy đủ** (bao gồm embedding + reranker):

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `sentence-transformers` và `faiss-cpu` là optional. Pipeline mặc định chạy với BM25 (không cần GPU). Chỉ cần cài khi muốn dùng dense embedding hoặc cross-encoder reranker.

---

## 4. Cấu trúc dữ liệu

Dữ liệu đã có sẵn trong folder `data/`:

```
data/
├── API_config_data/
│   └── Tài liệu config API.xlsx     # 131 APIs + 23 bảng alias
├── Document_config_data/
│   ├── Public_001.pdf                # 398 file PDF tài liệu
│   ├── Public_002.pdf
│   └── ...
├── example_data/
│   └── example_data.xlsx             # 100 câu hỏi mẫu + đáp án
└── test_data/
    └── Test_data.xlsx                # 617 câu hỏi test
```

**Không cần tải thêm dữ liệu.** Tất cả đã có sẵn.

---

## 5. Hướng dẫn chạy project

### ⚡ Chạy nhanh (3 bước)

```bash
# Bước 1: Build dữ liệu offline
python scripts/build_api_registry.py
python scripts/build_document_kb.py

# Bước 2: Train selector
python scripts/train_selector.py

# Bước 3: Chạy submission
python run_submission.py
```

Kết quả sẽ xuất ra tại:
- `outputs/predictions.jsonl` – Dự đoán dạng JSONL
- `outputs/submission.csv` – File nộp bài

---

### 📋 Hướng dẫn chi tiết từng bước

#### Bước 1: Build API Registry & Alias Dictionary

```bash
python scripts/build_api_registry.py
```

Script này sẽ:
- Đọc file `Tài liệu config API.xlsx`
- Trích xuất 131 API definitions → `data/processed/api_registry.jsonl`
- Trích xuất 23 bảng alias (organization, projectType, ...) → `data/processed/alias_dictionary.json`

**Output mong đợi:**
```
Loaded 131 APIs
Wrote API registry to data/processed/api_registry.jsonl
Wrote alias dictionary with 23 categories
  organization: 7 entries
  projectType: 4 entries
  project_info: 300 entries
  ...
```

#### Bước 2: Build Document Knowledge Base

```bash
python scripts/build_document_kb.py
```

Script này sẽ:
- Parse 398 file PDF bằng PyMuPDF (text-layer extraction)
- Chia thành chunks ngữ nghĩa (512 tokens, overlap 64)
- Xuất ra `data/processed/document_chunks.jsonl`

**Output mong đợi:**
```
Found 398 PDF files
Total chunks: 15539
Wrote document chunks to data/processed/document_chunks.jsonl
  Documents: 398
  Avg chunks/doc: 39.0
```

> ⏱️ Thời gian chạy: ~22 giây

#### Bước 3: Train Selector Model

```bash
python scripts/train_selector.py
```

Script này sẽ:
- Đọc 100 câu hỏi mẫu từ `example_data.xlsx`
- Train TF-IDF + LogisticRegression classifier
- Cross-validation 5-fold
- Lưu model → `data/cache/selector_model/`

**Output mong đợi:**
```
Training data: 100 samples
  call_api: 50
  call_document: 50
Cross-validation accuracy: 0.9800 (+/- 0.0245)
Training accuracy: 1.0000
Selector model saved
```

#### Bước 4: Đánh giá local (optional)

```bash
python scripts/evaluate_local.py
```

Chạy pipeline trên 100 câu hỏi mẫu và so sánh với đáp án:

**Output mong đợi:**
```
Routing accuracy: 100/100 = 1.0000
API path accuracy: 47/50 = 0.9400
Average time_response: 0.024s
```

#### Bước 5: Chạy submission trên test data

```bash
python run_submission.py
```

Chạy pipeline trên toàn bộ 617 câu hỏi test:

**Output mong đợi:**
```
=== Submission Summary ===
Total predictions: 617
  call_api: 348
  call_document: 269
  avg time_response: 0.023s
```

Kết quả được lưu tại:
- `outputs/predictions.jsonl`
- `outputs/submission.csv`

#### Bước 6: Test câu hỏi đơn lẻ (optional)

```bash
python src/main.py "Trong Quý 3/2025 thì TTPMTCS có bao nhiêu dự án"
```

---

## 6. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: id, question                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Normalize  │  Unicode, whitespace, time
                    │  Question   │  expressions
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Selector   │  TF-IDF + LogisticRegression
                    │  (98% CV)   │  question → call_api / call_document
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │ call_document│          │  call_api   │
       └──────┬──────┘          └──────┬──────┘
              │                         │
    ┌─────────▼─────────┐     ┌────────▼────────┐
    │ BM25 Retrieval    │     │ BM25 Retrieval  │
    │ (15,539 chunks)   │     │ (131 APIs)      │
    └─────────┬─────────┘     └────────┬────────┘
              │                         │
    ┌─────────▼─────────┐     ┌────────▼────────┐
    │ Parse A/B/C/D     │     │ Heuristic       │
    │ from note         │     │ Reranking       │
    └─────────┬─────────┘     └────────┬────────┘
              │                         │
    ┌─────────▼─────────┐     ┌────────▼────────┐
    │ Option Scoring    │     │ Slot Extraction │
    │ (keyword overlap) │     │ (date, org, ...) │
    └─────────┬─────────┘     └────────┬────────┘
              │                         │
              │                ┌────────▼────────┐
              │                │ Alias Normalize │
              │                │ (3-layer fuzzy) │
              │                └────────┬────────┘
              │                         │
              │                ┌────────▼────────┐
              │                │ Template Fill   │
              │                │ + Validate JSON │
              │                └────────┬────────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │   OUTPUT    │
                    │ id, func,   │
                    │ answer, time│
                    └─────────────┘
```

### Luồng xử lý chính

1. **Question Normalization** – Unicode NFKC, whitespace, time phrase chuẩn hóa
2. **Function Selection** – TF-IDF (word + char n-grams) + LogisticRegression
3. **call_document** – BM25 retrieval → parse options A/B/C/D → keyword overlap scoring → chọn đáp án
4. **call_api** – BM25 retrieval → heuristic rerank → slot extraction (date, org, type) → 3-layer alias normalization (exact → accent-insensitive → fuzzy) → template fill → JSON validation

---

## 7. Cấu trúc project

```
AI_kickoff/
│
├── README.md                   # ← File này
├── plan.md                     # Tài liệu thiết kế chi tiết
├── requirements.txt            # Dependencies
├── run_submission.py           # Entry point: chạy submission
│
├── configs/
│   ├── constants.py            # Hằng số: function codes, thresholds
│   ├── paths.py                # Đường dẫn tập trung
│   └── model_config.py         # Cấu hình model: embedding, reranker, LLM
│
├── src/
│   ├── main.py                 # Test câu hỏi đơn lẻ
│   ├── pipeline.py             # Pipeline inference chính
│   ├── schemas.py              # Data classes: InputSample, PredictionResult, APIEntry
│   │
│   ├── preprocess/
│   │   ├── question_normalizer.py  # Chuẩn hóa câu hỏi
│   │   ├── text_cleaner.py         # Làm sạch text
│   │   └── time_normalizer.py      # Parse biểu thức thời gian (T1/2025, Quý 3/2025)
│   │
│   ├── selector/
│   │   ├── feature_builder.py      # TF-IDF feature builder
│   │   ├── selector_model.py       # LogisticRegression classifier
│   │   └── predictor.py            # High-level selector predictor
│   │
│   ├── document/
│   │   ├── pdf_parser.py           # Parse PDF bằng PyMuPDF
│   │   ├── chunker.py              # Chia text thành chunks
│   │   ├── retriever.py            # Hybrid BM25 + dense retrieval
│   │   ├── option_parser.py        # Parse đáp án A/B/C/D từ note
│   │   ├── option_scorer.py        # Scoring options vs evidence
│   │   └── document_solver.py      # End-to-end document solver
│   │
│   ├── api/
│   │   ├── api_catalog_loader.py   # Load API từ Excel
│   │   ├── api_retriever.py        # BM25 retrieval cho API
│   │   ├── api_reranker.py         # Rerank API candidates
│   │   ├── slot_extractor.py       # Trích xuất slot (date, org, type)
│   │   ├── slot_normalizer.py      # 3-layer alias normalization
│   │   ├── template_filler.py      # Điền JSON template
│   │   ├── validator.py            # Validate output JSON
│   │   └── api_solver.py           # End-to-end API solver
│   │
│   ├── models/
│   │   ├── embedding_model.py      # Wrapper: sentence-transformers
│   │   ├── reranker_model.py       # Wrapper: cross-encoder
│   │   └── llm_wrapper.py          # Wrapper: LLM (vLLM/SGLang)
│   │
│   ├── output/
│   │   ├── formatter.py            # Format output theo schema cuộc thi
│   │   └── writer.py               # Ghi JSONL / CSV
│   │
│   └── utils/
│       ├── io_utils.py             # Đọc/ghi file (JSONL, JSON, Excel, CSV)
│       ├── json_utils.py           # Parse JSON robust
│       ├── timer.py                # Đo thời gian inference
│       ├── logging_utils.py        # Logging
│       └── fuzzy_match.py          # Fuzzy matching cho alias
│
├── scripts/
│   ├── build_api_registry.py       # Build API registry + alias dict
│   ├── build_document_kb.py        # Build document knowledge base
│   ├── build_alias_dict.py         # Build alias dictionary (standalone)
│   ├── train_selector.py           # Train branch selector
│   └── evaluate_local.py           # Đánh giá trên example data
│
├── evaluation/
│   ├── metrics.py                  # Metrics: routing, API path, doc answer accuracy
│   └── __init__.py
│
├── data/
│   ├── API_config_data/            # File Excel cấu hình API
│   ├── Document_config_data/       # 398 file PDF
│   ├── example_data/               # Câu hỏi mẫu + đáp án
│   ├── test_data/                  # Câu hỏi test
│   ├── processed/                  # [Generated] Dữ liệu đã xử lý
│   │   ├── api_registry.jsonl
│   │   ├── alias_dictionary.json
│   │   └── document_chunks.jsonl
│   └── cache/                      # [Generated] Model + index cache
│       └── selector_model/
│
└── outputs/                        # [Generated] Kết quả
    ├── predictions.jsonl
    └── submission.csv
```

---

## 8. Chi tiết từng module

### 8.1. Selector (`src/selector/`)

- **Thuật toán:** TF-IDF (word 1-3 gram + char 2-5 gram) → LogisticRegression
- **Input:** `question` (đã normalize)
- **Output:** `call_document` hoặc `call_api`
- **Hiệu suất:** 98% cross-validation, 100% training accuracy
- **Thời gian:** < 1ms

### 8.2. Document Pipeline (`src/document/`)

| Bước | Module | Mô tả |
|------|--------|-------|
| 1 | `pdf_parser.py` | Extract text từ PDF (PyMuPDF text-layer) |
| 2 | `chunker.py` | Chia thành chunks 512 tokens, overlap 64 |
| 3 | `retriever.py` | BM25 retrieval, hỗ trợ filter theo doc_id (Public_XXX) |
| 4 | `option_parser.py` | Parse A/B/C/D từ trường `note` |
| 5 | `option_scorer.py` | Keyword overlap scoring (fallback) hoặc cross-encoder |
| 6 | `document_solver.py` | Orchestrator: retrieve → parse → score → chọn đáp án |

### 8.3. API Pipeline (`src/api/`)

| Bước | Module | Mô tả |
|------|--------|-------|
| 1 | `api_catalog_loader.py` | Load 131 API + 23 bảng alias từ Excel |
| 2 | `api_retriever.py` | BM25 retrieval trên API registry |
| 3 | `api_reranker.py` | Heuristic rerank (description + example overlap) |
| 4 | `slot_extractor.py` | Extract: fromDate, toDate, organization, projectType, ... |
| 5 | `slot_normalizer.py` | 3-layer: exact → accent-insensitive → fuzzy (rapidfuzz) |
| 6 | `template_filler.py` | Điền slot vào JSON template |
| 7 | `validator.py` | Validate path, body params, enum values |
| 8 | `api_solver.py` | Orchestrator: retrieve → rerank → extract → normalize → fill |

### 8.4. Time Normalizer (`src/preprocess/time_normalizer.py`)

Hỗ trợ parse các dạng biểu thức thời gian tiếng Việt:

| Pattern | Ví dụ | fromDate | toDate |
|---------|-------|----------|--------|
| Tháng | T11/2025, tháng 8/2025 | 2025-11-01 | 2025-11-30 |
| Quý | Quý 3/2025, Q3/2025 | 2025-07-01 | 2025-09-30 |
| Năm | năm 2025 | 2025-01-01 | 2025-12-31 |
| Khoảng | T1/2025 - T12/2025 | 2025-01-01 | 2025-12-31 |

---

## 9. Cấu hình

### 9.1. Thay đổi model (`configs/model_config.py`)

```python
# Embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"          # Mặc định
# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"  # Accuracy-first

# Reranker model
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"     # Mặc định
# RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"    # Accuracy-first

# LLM (optional fallback)
LLM_MODEL_NAME = "Qwen/Qwen3-32B"
LLM_API_BASE = "http://localhost:8000/v1"  # vLLM endpoint
```

### 9.2. Thay đổi thresholds (`configs/constants.py`)

```python
DOC_RETRIEVAL_TOP_K = 50      # Số chunks retrieve cho document
API_RETRIEVAL_TOP_K = 30      # Số API candidates
API_RERANK_TOP_K = 5          # Số API sau rerank
CHUNK_SIZE = 512              # Kích thước chunk
CHUNK_OVERLAP = 64            # Overlap giữa chunks
```

### 9.3. Thay đổi đường dẫn data (`configs/paths.py`)

Tất cả đường dẫn được quản lý tập trung. Chỉ cần sửa nếu đổi vị trí data.

---

## 10. Kết quả đạt được

### Evaluation trên example data (100 câu)

| Metric | Giá trị |
|--------|---------|
| Routing accuracy | **100%** (100/100) |
| API path accuracy | **94%** (47/50) |
| Avg response time | **0.024s** |
| Cross-validation accuracy | **98% ± 2.5%** |

### Submission trên test data (617 câu)

| Metric | Giá trị |
|--------|---------|
| Total predictions | 617 |
| call_api | 348 (56.4%) |
| call_document | 269 (43.6%) |
| Avg response time | **0.023s** |

---

## 11. Nâng cấp nâng cao

### 11.1. Bật Dense Embedding (nâng cao retrieval)

```bash
pip install sentence-transformers
```

Model sẽ tự động sử dụng `BAAI/bge-m3` cho hybrid BM25 + dense retrieval.

### 11.2. Bật Cross-Encoder Reranker (nâng cao scoring)

Cross-encoder đã được tích hợp sẵn trong `sentence-transformers`. Cần khởi tạo `OptionScorer` và `APIReranker` với reranker model.

### 11.3. Bật LLM Fallback (xử lý trường hợp mơ hồ)

```bash
# Cài vLLM
pip install vllm

# Khởi động server
vllm serve Qwen/Qwen3-32B --port 8000

# Pipeline sẽ tự dùng LLM cho slot extraction mơ hồ
```

### 11.4. OCR cho PDF chất lượng thấp

```bash
pip install paddleocr paddlepaddle
```

Kích hoạt OCR fallback cho các trang PDF có `text_coverage < 0.1`.

---

## 12. Xử lý lỗi thường gặp

### Lỗi UnicodeEncodeError khi log tiếng Việt

```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Giải pháp:** Chạy với flag UTF-8:
```bash
python -X utf8 run_submission.py
```

Hoặc set biến môi trường:
```bash
set PYTHONIOENCODING=utf-8    # Windows
export PYTHONIOENCODING=utf-8  # Linux/macOS
```

> **Lưu ý:** Lỗi này chỉ ảnh hưởng log hiển thị, **không ảnh hưởng kết quả**. Pipeline vẫn chạy đúng.

### Lỗi ModuleNotFoundError

```bash
# Cài package bị thiếu
pip install <package-name>

# Hoặc cài toàn bộ
pip install -r requirements.txt
```

### Lỗi "Selector model not found"

Chạy train selector trước:
```bash
python scripts/train_selector.py
```

### Lỗi "Document index not found"

Chạy build document KB trước:
```bash
python scripts/build_document_kb.py
```

### Pipeline chạy chậm

- Đảm bảo dữ liệu processed đã được build (không cần build lại mỗi lần chạy)
- Giảm `DOC_RETRIEVAL_TOP_K` trong `configs/constants.py` nếu cần nhanh hơn

---

## Tác giả

Đội thi Viettel AI Race

## License

MIT
