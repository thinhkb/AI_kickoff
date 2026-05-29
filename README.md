# Viettel AI Race - Smart Chatbot for RAG and API Integration

This project implements a two-branch chatbot pipeline for the Viettel AI Race.
It decides whether a question should be answered from document evidence or
translated into an API request, then returns the competition-required output
format.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Quick Start](#quick-start)
- [Detailed Workflow](#detailed-workflow)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Module Guide](#module-guide)
- [Configuration](#configuration)
- [Problem Solver](#problem-solver)
- [Results](#results)
- [Advanced Options](#advanced-options)
- [Troubleshooting](#troubleshooting)

## Overview

The system supports two function codes:

| Function code | Purpose | Input | Output |
| --- | --- | --- | --- |
| `call_document` | Answer multiple-choice questions from PDF-derived evidence. | `question` + `note` with A/B/C/D options | `{"numbers": 1, "result": "A"}` |
| `call_api` | Select the correct API and fill its JSON body. | `question` | `{"path": "/api/...", "body": {...}}` |

Core design principles:

- Use the question as the main routing signal.
- Keep the two branches explicit: document questions are solved by evidence
  retrieval and option scoring; API questions are solved by API retrieval,
  slot extraction, normalization, template filling, and validation.
- Prefer fast deterministic logic where possible, with optional Qwen3 reranking
  for harder cases.
- Build processed data once, then reuse cached artifacts for fast submission.

## System Requirements

| Component | Minimum |
| --- | --- |
| Python | 3.10+ |
| RAM | 8 GB+ |
| Disk | 2 GB+ for data and cached artifacts |
| GPU | Optional; useful only when enabling the Qwen3 reranker |
| OS | Windows, Linux, or macOS |

## Installation

Clone the project and create a virtual environment:

```bash
git clone <repo-url>
cd AI_kickoff

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For a lightweight CPU-only run, the essential packages are:

```bash
pip install openpyxl pandas scikit-learn numpy scipy rapidfuzz rank-bm25 tqdm PyMuPDF joblib jsonschema
```

`sentence-transformers` is needed only when you enable the optional Qwen3
reranker. The default fast path can run with lexical retrieval and heuristic
scoring.

## Data Layout

The expected data is stored under `data/`:

```text
data/
  API_config_data/          API configuration workbook
  Document_config_data/     Source PDF documents
  example_data/             Example questions and labels
  test_data/                Test questions
  processed/                Generated JSON/JSONL artifacts
  cache/                    Generated model/index cache
```

Important generated artifacts:

| File | Built by | Purpose |
| --- | --- | --- |
| `data/processed/api_registry.jsonl` | `scripts/build_api_registry.py` | Normalized API catalog for retrieval. |
| `data/processed/alias_dictionary.json` | `scripts/build_api_registry.py` or `scripts/build_alias_dict.py` | Alias tables for organizations, project types, statuses, and other API slots. |
| `data/processed/document_chunks.jsonl` | `scripts/build_document_kb.py` | Document chunks used by the document retriever. |
| `data/cache/selector_model/` | `scripts/train_selector.py` | Trained TF-IDF + Logistic Regression branch selector. |
| `outputs/predictions*.jsonl` | `run_submission.py` | Full prediction records. |
| `outputs/submission*.csv` | `run_submission.py` | Competition submission file. |

## Quick Start

Run these commands from the project root:

```bash
python scripts/build_api_registry.py
python scripts/build_document_kb.py
python scripts/train_selector.py
python run_submission.py
```

The submission outputs are written to `outputs/`.

## Detailed Workflow

### 1. Build the API registry

```bash
python scripts/build_api_registry.py
```

This reads the API workbook, extracts API definitions, and writes:

- `data/processed/api_registry.jsonl`
- `data/processed/alias_dictionary.json`

### 2. Build the document knowledge base

```bash
python scripts/build_document_kb.py
```

This parses the PDF or MinerU Markdown sources, chunks the content, and writes:

- `data/processed/document_chunks.jsonl`

If MinerU Markdown files are available, the pipeline uses the MinerU-aware
chunker. Otherwise it falls back to PyMuPDF text extraction.

### 3. Train the selector

```bash
python scripts/train_selector.py
```

The selector learns whether a normalized question should go to:

- `call_document`
- `call_api`

The trained model is saved under `data/cache/selector_model/`.

### 4. Evaluate locally

```bash
python scripts/evaluate_local.py
```

This runs the pipeline on `example_data.xlsx` and reports routing accuracy,
API path accuracy, and response time.

### 5. Run submission

```bash
python run_submission.py
```

This processes the full test workbook and writes versioned prediction and
submission files under `outputs/`.

### 6. Run a single question

```bash
python src/main.py "Trong Quy 3/2025 thi TTPMTCS co bao nhieu du an?"
```

## Architecture

```text
Input sample
  id, question, note
        |
        v
Question normalization
  Unicode cleanup, whitespace cleanup, time expression support
        |
        v
Branch selector
  TF-IDF word/char n-grams + Logistic Regression
        |
        +------------------------------+
        |                              |
        v                              v
call_document                     call_api
  document retrieval                API retrieval
  option parsing                    heuristic or Qwen3 reranking
  option scoring                    slot extraction
  answer selection                  alias normalization
                                   template filling
                                   JSON validation
        |                              |
        +---------------+--------------+
                        v
Competition output
  id, function_code, function_answer, time_response
```

Main flow:

1. `src/preprocess/question_normalizer.py` normalizes question text.
2. `src/selector/selector_model.py` routes the question to one of the two
   branches.
3. `src/document/document_solver.py` handles PDF/document multiple-choice
   questions.
4. `src/api/api_solver.py` handles API selection and JSON body generation.
5. `src/output/writer.py` writes JSONL and CSV outputs.

## Project Structure

```text
AI_kickoff/
  README.md
  requirements.txt
  run_submission.py

  configs/
    constants.py          Function codes, thresholds, chunk sizes.
    paths.py              Centralized data, cache, and output paths.
    model_config.py       Qwen3 model names and selector settings.

  src/
    main.py               Single-question entry point.
    pipeline.py           End-to-end inference pipeline.
    schemas.py            Dataclasses for samples, predictions, chunks, APIs.

    preprocess/           Question, text, and time normalization.
    selector/             TF-IDF feature builder and branch selector.
    document/             PDF parsing, chunking, retrieval, option scoring.
    api/                  API catalog, retrieval, slot extraction, filling.
    models/               Qwen3 embedding, reranker, and LLM wrappers.
    output/               Competition output formatting and writing.
    utils/                IO, JSON, logging, timing, and fuzzy matching.

  scripts/
    build_api_registry.py
    build_alias_dict.py
    build_document_kb.py
    train_selector.py
    evaluate_local.py
    mineru_batch_extract.py
    debug_*.py

  evaluation/
    metrics.py

  data/
    API_config_data/
    Document_config_data/
    example_data/
    test_data/
    processed/
    cache/

  outputs/
```

## Module Guide

### Selector

Location: `src/selector/`

- Uses TF-IDF word n-grams and character n-grams.
- Uses Logistic Regression for fast and stable routing.
- Has a heuristic fallback when the trained selector is missing.
- Is protected by routing safety rules in `src/pipeline.py`, for example:
  if a row has multiple-choice options in `note`, it is treated as a document
  question.

### Document Branch

Location: `src/document/`

| Step | Module | Responsibility |
| --- | --- | --- |
| Parse | `pdf_parser.py`, `mineru_parser.py` | Extract text or structured Markdown from source documents. |
| Chunk | `chunker.py`, `mineru_chunker.py` | Split documents into retrievable chunks. |
| Retrieve | `retriever.py` | BM25 retrieval with document-id filtering. |
| Parse options | `option_parser.py` | Extract A/B/C/D answer choices from `note`. |
| Score | `option_scorer.py` | Score options against retrieved evidence. |
| Solve | `document_solver.py` | Orchestrate retrieval, scoring, and answer selection. |

### API Branch

Location: `src/api/`

| Step | Module | Responsibility |
| --- | --- | --- |
| Load catalog | `api_catalog_loader.py` | Read API definitions and alias dictionaries. |
| Retrieve | `api_retriever.py` | Use BM25 to find candidate APIs. |
| Rerank | `api_reranker.py` | Use heuristic scoring or optional Qwen3 reranker. |
| Extract slots | `slot_extractor.py` | Extract dates, organizations, statuses, project types, and other parameters. |
| Normalize slots | `slot_normalizer.py` | Map aliases, abbreviations, and misspellings to backend values. |
| Fill body | `template_filler.py` | Build the JSON body from calibrated templates or endpoint schema. |
| Validate | `validator.py` | Validate path, required fields, and output consistency. |
| Solve | `api_solver.py` | Orchestrate API retrieval, reranking, extraction, normalization, and filling. |

## Configuration

### Model configuration

The project is configured for Qwen3 models in `configs/model_config.py`.

```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MAX_SEQ_LENGTH = 8192
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_NORMALIZE = True

RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
RERANKER_MAX_SEQ_LENGTH = 1024
RERANKER_BATCH_SIZE = 16

LLM_MODEL_NAME = "Qwen/Qwen3-32B"
LLM_API_BASE = "http://localhost:8000/v1"
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.0
```

The heavy reranker is disabled by default to keep the submission pipeline fast.
Enable it only when needed:

```bash
# Windows PowerShell
$env:VIETTEL_USE_RERANKER="1"
$env:VIETTEL_RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"

# Linux / macOS
export VIETTEL_USE_RERANKER=1
export VIETTEL_RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"
```

Optional Qwen3 LLM serving:

```bash
vllm serve Qwen/Qwen3-32B --port 8000
```

### Retrieval and chunking thresholds

Edit `configs/constants.py`:

```python
DOC_RETRIEVAL_TOP_K = 50
DOC_RERANK_TOP_K = 10
API_RETRIEVAL_TOP_K = 30
API_RERANK_TOP_K = 5

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SELECTOR_THRESHOLD = 0.5
MAX_TIME_RESPONSE = 15.0
```

### Paths

All data, cache, processed, and output paths are centralized in
`configs/paths.py`. Change paths there if the folder layout changes.

## Problem Solver

This section summarizes the practical problems found during implementation and
how the project handles them.

### 1. Abbreviations and internal codes

Problem: Competition questions often use short internal names or procurement
codes such as organization abbreviations, project types, or terms like `DTRR`,
`DTHC`, `CHCT`, `MSTT`, `T&M`, `ODC`, and `presales`.

Solution:

- `src/api/slot_extractor.py` contains deterministic extraction patterns for
  common project types, statuses, procurement aliases, and organization tokens.
- `data/processed/alias_dictionary.json` stores workbook-derived aliases.
- `src/api/slot_normalizer.py` maps extracted text to backend values before
  template filling.

### 2. Typos, missing accents, and inconsistent Vietnamese input

Problem: User questions may contain Vietnamese without accents, mistyped
organization names, inconsistent casing, smart quotes, dash variants, or extra
spaces.

Solution:

- `src/preprocess/text_cleaner.py` normalizes Unicode, quotes, dashes,
  whitespace, and punctuation spacing.
- `src/utils/fuzzy_match.py` supports exact matching, accent-insensitive
  matching, and fuzzy matching with RapidFuzz.
- `src/api/slot_normalizer.py` applies matching in layers:
  exact -> accent-insensitive -> fuzzy.

This lets values such as organization names or project-type aliases survive
minor spelling differences before being sent to the API body.

### 3. Ambiguous routing between document and API questions

Problem: Some API questions look like document questions because both may ask
for counts, dates, or project information. Some document questions also look
API-like because they contain numeric/table references.

Solution:

- The selector uses both word n-grams and character n-grams, which helps with
  short Vietnamese terms and abbreviations.
- `src/pipeline.py` adds routing safety rules:
  - if `note` is empty and the selector chose `call_document`, route to
    `call_api`;
  - if `note` contains multiple-choice options and the selector chose
    `call_api`, route back to `call_document`.

### 4. Time expressions in Vietnamese questions

Problem: API bodies often need `fromDate` and `toDate`, but questions use many
forms: `T1/2025`, `thang 8/2025`, `Quy 3/2025`, `Q3/2025`, `nam 2025`, or
ranges such as `T1/2025 - T12/2025`.

Solution:

- `src/preprocess/time_normalizer.py` extracts month, quarter, year, and range
  expressions.
- `src/api/slot_extractor.py` reuses `extract_date_range()` to fill API date
  slots in `YYYY-MM-DD` format.

### 5. Similar API endpoints

Problem: Several APIs have very similar descriptions and body schemas. A pure
keyword match can select the wrong endpoint.

Solution:

- `src/api/api_retriever.py` retrieves a broad candidate set with BM25.
- `src/api/api_reranker.py` reranks using retrieval score, description overlap,
  example-question overlap, path-specific keyword boosts, or optional Qwen3
  reranking.
- `src/api/template_filler.py` uses calibrated templates from example answers
  when available, reducing output drift.

### 6. API body shape and default values

Problem: Even when the correct API path is selected, the JSON body can fail if
list fields, booleans, integers, or optional fields use the wrong default.

Solution:

- `src/api/template_filler.py` builds neutral defaults by parameter type and
  endpoint family.
- It preserves calibrated body key order and expected default types.
- `src/api/validator.py` checks the generated API output before it is returned.

### 7. Document ID hints and evidence retrieval

Problem: Document questions may mention `Public_123`, `Public 123`, or `TD123`.
Without document filtering, retrieval can pick evidence from a different PDF.

Solution:

- `src/document/retriever.py` detects document IDs and filters retrieval to the
  matching document when possible.
- If document-specific retrieval is too sparse, it falls back to leading chunks
  from the same document instead of returning no evidence.

### 8. Multiple-choice option scoring and negation

Problem: For document questions, the correct option is not always the option
with the most obvious keyword overlap. Negative questions such as "which one is
not correct" also change the interpretation.

Solution:

- `src/document/option_parser.py` extracts stable A/B/C/D choices from `note`.
- `src/document/option_scorer.py` combines normalized overlap, proximity,
  BM25 retrieval for question + option, option-only retrieval, TF-IDF cosine
  similarity, and optional Qwen3 reranking.
- The scorer includes Vietnamese negation patterns so negative questions can
  be handled more carefully.

### 9. Speed versus accuracy

Problem: The submission pipeline must process many rows quickly, so using a
large model for every step would be too slow.

Solution:

- Build scripts precompute reusable artifacts.
- Runtime defaults use BM25, TF-IDF, deterministic rules, and cached selector
  models.
- Qwen3 reranking is optional and controlled by environment variables.

## Results

Local example-data results documented during development:

| Metric | Value |
| --- | --- |
| Routing accuracy | 100/100 |
| API path accuracy | 47/50 |
| Average response time | around 0.024 seconds |
| Selector cross-validation | around 98% |

Submission run summary documented during development:

| Metric | Value |
| --- | --- |
| Total predictions | 617 |
| `call_api` | 348 |
| `call_document` | 269 |
| Average response time | around 0.023 seconds |

## Advanced Options

### Optional Qwen3 reranker

Install `sentence-transformers`, then enable the reranker with environment
variables:

```bash
pip install sentence-transformers

# Windows PowerShell
$env:VIETTEL_USE_RERANKER="1"
$env:VIETTEL_RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"

# Linux / macOS
export VIETTEL_USE_RERANKER=1
export VIETTEL_RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"
```

### Optional Qwen3 LLM endpoint

```bash
pip install vllm
vllm serve Qwen/Qwen3-32B --port 8000
```

The LLM endpoint is configured in `configs/model_config.py`.

### MinerU document extraction

MinerU can be used offline for PDFs with complex layouts, tables, formulas, or
poor text-layer quality.

Recommended flow:

```bash
# Use a separate environment if needed.
python -m venv .mineru_venv
.mineru_venv\Scripts\activate

pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com

python scripts/mineru_batch_extract.py
```

After MinerU extraction, return to the main project environment and rebuild the
document knowledge base:

```bash
python scripts/build_document_kb.py
```

If MinerU Markdown exists, `build_document_kb.py` uses the MinerU chunker. If
not, it falls back to PyMuPDF.

## Troubleshooting

### UnicodeEncodeError when logging Vietnamese text

```text
UnicodeEncodeError: 'charmap' codec can't encode character
```

Run Python in UTF-8 mode:

```bash
python -X utf8 run_submission.py
```

Or set the environment variable:

```bash
# Windows
set PYTHONIOENCODING=utf-8

# Linux / macOS
export PYTHONIOENCODING=utf-8
```

### ModuleNotFoundError

Install the missing package or reinstall all dependencies:

```bash
pip install <package-name>
pip install -r requirements.txt
```

### Selector model not found

Train the selector:

```bash
python scripts/train_selector.py
```

### Document index not found

Build the document knowledge base:

```bash
python scripts/build_document_kb.py
```

### API registry not found

Build the API registry:

```bash
python scripts/build_api_registry.py
```

### Pipeline is slow

- Make sure processed data has already been built.
- Keep `VIETTEL_USE_RERANKER` disabled for fast submissions.
- Reduce `DOC_RETRIEVAL_TOP_K` or `API_RETRIEVAL_TOP_K` only if speed is more
  important than accuracy.

## Author

Viettel AI Race team

## License

MIT
