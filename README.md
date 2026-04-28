# AI_kickoff

# Viettel AI Race - Smart Chatbot for RAG & API Integration

## 1. Project Overview

This project implements a **two-branch intelligent chatbot** for the Viettel AI Race challenge.

The system solves two tasks:

- **Document Extraction (`call_document`)**  
  Answer multiple-choice questions from a knowledge base of PDF documents containing text, images, formulas, and tables.

- **API Configuration Generation (`call_api`)**  
  Select the correct API and fill the configuration JSON from a natural-language question using API specification files and backend-standardized parameter aliases.

The input only uses:

- `id`
- `question`

The `note` field is **not used during function routing**. It is only accessed **after** the system has already selected `function_code = "call_document"` in order to parse answer options A/B/C/D.

The output format is:

- `id`
- `function_code`
- `function_answer`
- `time_response`

---

## 2. Core Design Principles

Our solution is built around the following rules:

1. **No fallback routing**  
   The system does not default to `call_document` and then switch to another branch if retrieval fails.  
   It also does not default to `call_api` and retry elsewhere.

2. **No shortcut routing by answer format**  
   The system does not use the presence of options A/B/C/D as a hard rule to choose `call_document`.

3. **Direct function selection from the question**  
   A dedicated selector predicts either:
   - `call_document`
   - `call_api`

4. **Data-driven API handling**  
   We do **not** manually re-code each API as a separate business function.  
   Instead, we build an **API registry** from the provided API configuration file, then:
   - retrieve the correct API
   - extract parameters from the question
   - normalize aliases / abbreviations / typos
   - generate the final JSON output

5. **Fast inference-first engineering**  
   Since runtime is graded, the system minimizes unnecessary LLM generation and uses:
   - retrieval
   - reranking
   - structured parsing
   - deterministic template filling

---

## 3. High-Level Workflow

flowchart TD
    A[Input: id, question] --> B[Question Normalization]
    B --> C[Function Selector]

    C -->|call_document| D[Document Pipeline]
    C -->|call_api| E[API Pipeline]

    D --> D1[Retrieve relevant PDF chunks]
    D1 --> D2[Read note and parse A/B/C/D]
    D2 --> D3[Score options against retrieved evidence]
    D3 --> D4[Return selected answer(s)]

    E --> E1[Retrieve top API candidates]
    E1 --> E2[Extract slots from question]
    E2 --> E3[Normalize aliases, abbreviations, typos]
    E3 --> E4[Fill API JSON template]
    E4 --> E5[Validate output schema]

    D4 --> F[Build final output]
    E5 --> F[Build final output]
    F --> G[Output: id, function_code, function_answer, time_response]
````

---

## 4. End-to-End Pipeline

### 4.1 Offline Preparation

Offline steps are executed before inference and are **not counted** in `time_response`.

#### A. Document Knowledge Base

We preprocess all PDF files into a searchable document index:

* extract text blocks
* preserve page number and section metadata
* separate tables when possible
* optionally OCR pages with poor text extraction
* chunk the content into semantically meaningful segments

#### B. API Knowledge Base

We convert the API configuration Excel into a structured API registry:

* `func_code`
* API name
* description
* example question
* request path / method
* required parameters
* optional parameters
* expected JSON schema

#### C. Alias Dictionary

We build a normalization dictionary from backend-standardized aliases:

* abbreviations
* typo-tolerant variants
* accent-insensitive matching
* fuzzy string matching support

#### D. Selector Training / Calibration

Using the provided example data, we train or calibrate a lightweight classifier that predicts:

* `call_document`
* `call_api`

The selector only uses the `question`.

---

### 4.2 Online Inference

#### Step 1. Question Normalization

Input question is normalized with:

* unicode normalization
* whitespace cleanup
* lowercase copy
* time phrase normalization
* typo-tolerant preprocessing

#### Step 2. Function Selection

A dedicated selector predicts the execution branch directly:

* `call_document`
* `call_api`

This decision is based on semantic relevance and learned routing signals, not on fallback logic.

#### Step 3A. Document Branch (`call_document`)

The document branch performs:

1. retrieve relevant chunks from PDF knowledge base
2. read `note` only after routing is done
3. parse options A/B/C/D
4. score each option against retrieved evidence
5. return the best option or multiple options if needed

#### Step 3B. API Branch (`call_api`)

The API branch performs:

1. retrieve top API candidates from API registry
2. rerank the best candidates
3. extract entities / slots from the question
4. normalize aliases, abbreviations, and misspellings
5. fill the correct JSON template
6. validate output format

#### Step 4. Output Formatting

The final output always follows the challenge schema:

* `id`
* `function_code`
* `function_answer`
* `time_response`

---

## 5. Branch-Specific Strategy

## 5.1 `call_document`: Retrieval + Option Scoring

Instead of generating a long free-form answer, the document branch is optimized for multiple-choice QA:

* retrieve evidence from PDFs
* compare each candidate option with the evidence
* select the most supported answer

This approach is:

* faster
* more stable
* less hallucination-prone
* better aligned with objective evaluation

This is especially important because the sample public PDF contains technical content with hierarchical sections, code-like snippets, and images, so preserving document structure improves retrieval quality. 

---

## 5.2 `call_api`: API Retrieval + Slot Filling

The API branch is **not** implemented as hundreds of handwritten functions.

Instead, each API is treated as a structured template:

* choose the correct `func_code`
* identify needed parameters
* normalize slot values
* render the final JSON

This branch is designed as:

* API retrieval
* slot extraction
* alias normalization
* template filling
* schema validation

This makes the system:

* scalable
* easier to maintain
* faster to iterate
* robust to abbreviations and noisy queries

---

## 6. Handling Abbreviations, Typos, and Aliases

One key difficulty of `call_api` is that questions may contain:

* abbreviations
* backend aliases
* misspellings
* inconsistent naming

We handle them through a three-layer normalization strategy:

### Layer 1. Exact normalization

* lowercase normalization
* trim spaces
* accent-insensitive comparison
* direct alias dictionary lookup

### Layer 2. Fuzzy normalization

* edit distance / fuzzy matching
* typo correction
* approximate alias resolution

### Layer 3. Semantic normalization

* embedding-based similarity for unresolved terms
* canonical value selection among valid backend values

This normalization is separated from API selection to keep the pipeline modular and easier to debug.

---

## 7. Why This Architecture Fits the Challenge

This architecture is chosen because it matches the official constraints:

* It uses only `question` for branch selection.
* It does not use `note` before selecting `call_document`.
* It does not rely on fallback design.
* It does not hard-code multiple-choice logic for routing.
* It explicitly addresses API abbreviations and misspellings.
* It prioritizes runtime speed by limiting unnecessary generation.

---

## 8. Project Structure

```text
viettel-ai-race-chatbot/
│
├── README.md
├── requirements.txt
├── colab_demo.ipynb
├── run_submission.py
│
├── configs/
│   ├── constants.py
│   ├── paths.py
│   └── model_config.py
│
├── data/
│   ├── raw/
│   │   ├── pdfs/
│   │   ├── example_data.xlsx
│   │   ├── test_data.xlsx
│   │   └── api_config.xlsx
│   │
│   ├── processed/
│   │   ├── document_chunks.jsonl
│   │   ├── api_registry.jsonl
│   │   ├── alias_dictionary.json
│   │   └── selector_train.jsonl
│   │
│   └── cache/
│       ├── doc_index/
│       ├── api_index/
│       └── embeddings/
│
├── src/
│   ├── main.py
│   ├── pipeline.py
│   ├── schemas.py
│   │
│   ├── preprocess/
│   │   ├── question_normalizer.py
│   │   ├── text_cleaner.py
│   │   └── time_normalizer.py
│   │
│   ├── selector/
│   │   ├── feature_builder.py
│   │   ├── selector_model.py
│   │   └── predictor.py
│   │
│   ├── document/
│   │   ├── pdf_parser.py
│   │   ├── ocr_parser.py
│   │   ├── table_parser.py
│   │   ├── chunker.py
│   │   ├── retriever.py
│   │   ├── option_parser.py
│   │   ├── option_scorer.py
│   │   └── document_solver.py
│   │
│   ├── api/
│   │   ├── api_catalog_loader.py
│   │   ├── api_retriever.py
│   │   ├── api_reranker.py
│   │   ├── slot_extractor.py
│   │   ├── slot_normalizer.py
│   │   ├── template_filler.py
│   │   ├── validator.py
│   │   └── api_solver.py
│   │
│   ├── models/
│   │   ├── embedding_model.py
│   │   ├── reranker_model.py
│   │   └── llm_wrapper.py
│   │
│   ├── utils/
│   │   ├── io_utils.py
│   │   ├── json_utils.py
│   │   ├── timer.py
│   │   ├── logging_utils.py
│   │   └── fuzzy_match.py
│   │
│   └── output/
│       ├── formatter.py
│       └── writer.py
│
├── scripts/
│   ├── build_document_kb.py
│   ├── build_api_registry.py
│   ├── build_alias_dict.py
│   ├── train_selector.py
│   └── evaluate_local.py
│
├── evaluation/
│   ├── eval_selector.py
│   ├── eval_document.py
│   ├── eval_api.py
│   ├── metrics.py
│   └── error_analysis.py
│
└── outputs/
    ├── predictions.jsonl
    └── submission.csv
```

---

## 9. File-by-File Responsibilities

### Root Files

#### `README.md`

Project documentation, architecture overview, and usage instructions.

#### `requirements.txt`

List of all Python dependencies required to run the project.

#### `colab_demo.ipynb`

Google Colab notebook for end-to-end execution:

* install dependencies
* load data
* build indexes
* run inference
* export submission

#### `run_submission.py`

Batch prediction entry point for test-time inference and submission generation.

---

### `configs/`

#### `configs/constants.py`

Stores shared constants:

* function codes
* schema keys
* threshold values
* default retrieval settings

#### `configs/paths.py`

Centralizes all file and folder paths.

#### `configs/model_config.py`

Contains model names, checkpoint paths, token limits, and inference configuration.

---

### `src/preprocess/`

#### `question_normalizer.py`

Normalizes the input question before routing:

* casing
* punctuation
* spacing
* standard patterns

#### `text_cleaner.py`

Utility functions for text cleanup and canonicalization.

#### `time_normalizer.py`

Parses and standardizes time expressions such as:

* month / quarter / year
* relative date ranges
* compact date notations

---

### `src/selector/`

#### `feature_builder.py`

Builds selector features from the question and retrieval signals.

#### `selector_model.py`

Implements or loads the classifier that predicts:

* `call_document`
* `call_api`

#### `predictor.py`

High-level function for selector inference.

---

### `src/document/`

#### `pdf_parser.py`

Extracts text blocks and page metadata from PDF documents.

#### `ocr_parser.py`

Handles OCR for low-text or image-heavy pages when needed.

#### `table_parser.py`

Extracts or linearizes tables into retrieval-friendly text.

#### `chunker.py`

Splits parsed PDF content into semantic chunks with metadata.

#### `retriever.py`

Searches relevant document chunks for a given question.

#### `option_parser.py`

Parses A/B/C/D answer options from the `note` field.

#### `option_scorer.py`

Scores candidate options against retrieved document evidence.

#### `document_solver.py`

End-to-end solver for the `call_document` branch.

---

### `src/api/`

#### `api_catalog_loader.py`

Loads API definitions from Excel and converts them into a structured registry.

#### `api_retriever.py`

Retrieves top API candidates from the API registry.

#### `api_reranker.py`

Reranks top candidate APIs for more accurate selection.

#### `slot_extractor.py`

Extracts required entities and parameter values from the question.

#### `slot_normalizer.py`

Normalizes:

* aliases
* abbreviations
* misspellings
* backend-specific values

#### `template_filler.py`

Fills the selected API JSON template with normalized slot values.

#### `validator.py`

Validates the generated API output:

* field names
* required keys
* enum compatibility
* JSON schema consistency

#### `api_solver.py`

End-to-end solver for the `call_api` branch.

---

### `src/models/`

#### `embedding_model.py`

Wrapper for embedding model used in retrieval.

#### `reranker_model.py`

Wrapper for reranking model used in API and document candidate scoring.

#### `llm_wrapper.py`

Lightweight wrapper around the local model when generation or semantic scoring is needed.

---

### `src/utils/`

#### `io_utils.py`

File reading / writing helpers.

#### `json_utils.py`

JSON serialization, parsing, and formatting helpers.

#### `timer.py`

Measures inference time for `time_response`.

#### `logging_utils.py`

Logging utilities for debugging and tracing pipeline decisions.

#### `fuzzy_match.py`

Reusable fuzzy matching utilities for alias normalization.

---

### `src/output/`

#### `formatter.py`

Formats the final output into challenge-compliant schema.

#### `writer.py`

Writes batch predictions to JSONL / CSV submission file.

---

### `scripts/`

#### `build_document_kb.py`

Builds the document knowledge base and retrieval index.

#### `build_api_registry.py`

Builds the API registry from the API configuration file.

#### `build_alias_dict.py`

Builds alias dictionary and normalization resources.

#### `train_selector.py`

Trains or calibrates the branch selector using example data.

#### `evaluate_local.py`

Runs local evaluation before submission.

---

### `evaluation/`

#### `eval_selector.py`

Evaluates routing accuracy.

#### `eval_document.py`

Evaluates multiple-choice performance on the document branch.

#### `eval_api.py`

Evaluates API selection and JSON correctness.

#### `metrics.py`

Defines project-wide evaluation metrics.

#### `error_analysis.py`

Analyzes failure cases for debugging and iteration.

---

## 10. Main Inference Logic

The system can be summarized by the following pseudocode:

```python
def predict_one(sample):
    start_time = now()

    q = normalize_question(sample["question"])

    function_code = selector.predict(q)

    if function_code == "call_document":
        result = document_solver.solve(
            question=q,
            note=sample.get("note", "")
        )
    elif function_code == "call_api":
        result = api_solver.solve(question=q)
    else:
        raise ValueError("Invalid function_code")

    elapsed = now() - start_time

    return {
        "id": sample["id"],
        "function_code": function_code,
        "function_answer": result,
        "time_response": elapsed
    }
```

---

## 11. Why Our Team Chose This Solution

Our team selected this architecture for three reasons:

### A. Compliance with challenge constraints

The pipeline strictly respects:

* input field rules
* routing rules
* note usage rules
* no-fallback requirement

### B. Speed

We minimize heavy free-form generation and use structured retrieval + scoring + template filling.

### C. Robustness

The system is modular, debuggable, and specifically designed to handle:

* PDF knowledge extraction
* API alias normalization
* noisy real-world questions

---

## 12. Future Improvements

Potential future upgrades include:

* better OCR fallback for formula-heavy PDFs
* stronger reranker for difficult API selection cases
* improved time parser for ambiguous date expressions
* richer table understanding for document QA
* confidence calibration for selector and option scoring

---

## 13. Summary

This project is a **modular dual-branch chatbot system** for:

* document-based multiple-choice QA
* API configuration generation

The system is designed to be:

* competition-compliant
* fast
* accurate
* scalable
* easy to debug and extend

Instead of relying on a single large prompt, we combine:

* branch selection
* structured retrieval
* option scoring
* alias normalization
* deterministic template filling

This gives us a strong balance between **accuracy** and **response speed**, which are both central to the competition scoring.

