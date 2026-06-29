# 🏥 Clinical Trial RAG Pipeline
**Regulatory-grade document intelligence for clinical trial protocols.**  
Hybrid semantic + keyword retrieval · Multi-agent audit · Confidence-scored answers · Quantitative evaluation

---

## What this does

Clinical trial protocols are dense, table-heavy documents where a single misread number — a performance score threshold, a washout period, a contraceptive requirement — can delay a regulatory submission by months.

This pipeline turns a raw protocol PDF into a queryable, auditable knowledge system:

| Property | Implementation |
|---|---|
| Accuracy | PubMedBERT embeddings understand medical synonyms |
| Completeness | Hybrid BM25 + semantic retrieval catches what either alone misses |
| Traceability | Every answer is linked back to source pages |
| Safety | Deterministic audit layer flags threshold violations before outputs are trusted |

---

## Architecture

```
Protocol PDF
     │
     ├─► PyPDFLoader (prose sections)
     └─► UnstructuredPDFLoader (table-aware, §6.1 pages)
               │
               ▼
        Text Chunking (1200 chars, 200 overlap)
               │
     ┌─────────┴──────────┐
     │                    │
BM25 Retriever       FAISS Vector Store
(exact match)        (PubMedBERT semantic)
     │                    │
     └─────────┬──────────┘
               │  Hybrid Merge (deduplicated)
               ▼
    ┌──────────────────────┐
    │   Prompt Layer       │
    │  · Clinical QA       │
    │  · JSON Extraction   │
    │  · Ambiguity Detect  │
    │  · Confidence Score  │
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │   Multi-Agent Audit  │
    │  Agent 2: Rulebook   │  ← deterministic (fast, reproducible)
    │  Agent 3: LLM Review │  ← semantic verification
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │  RAGAS Evaluation    │
    │  · Faithfulness      │
    │  · Context Precision │
    └──────────────────────┘
```

---

## Project Structure

```
clinical-trial-rag/
├── main.py                      ← run this for the full interactive pipeline
├── src/
│   ├── prompts.py               ← all prompt templates (7 roles)
│   ├── ingestion.py             ← PDF loading + chunking + vector store
│   ├── retrieval.py             ← HybridRetriever class
│   └── evaluation.py           ← RegulatoryAuditor + RAGAS scoring
├── data/
│   ├── Protocol.pdf             ← place your protocol PDF here
│   ├── cache/                   ← auto-generated cache (gitignored)
│   └── sample_output/          ← all pipeline outputs saved here
│       └── example_extraction.json  ← sample output (synthetic data)
├── .env
├── .gitignore
└── requirements.txt
```

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/hardip11/clinical-trial-rag.git
cd clinical-trial-rag
pip install -r requirements.txt
```

### 2. Set your API token
```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
# Get one at https://huggingface.co/settings/tokens
```

### 3. Add your protocol PDF
Place your clinical trial protocol PDF at `data/Protocol.pdf`.

### 4. Run the pipeline
```bash
python main.py
```

Runs all 4 phases automatically, then enters an interactive Q&A session. All outputs are saved to `data/sample_output/`. Type `quit` to exit the Q&A session.

---

## What `python main.py` does

**Automatic phases:**
1. Ingests the protocol PDF via dual loaders (standard + table-aware)
2. Extracts structured eligibility criteria as validated JSON
3. Runs two audit agents — deterministic rulebook + LLM second-pass
4. Scans for regulatory risk flags (ambiguous language, missing specificity)
5. Runs a 4-query evaluation suite scoring faithfulness + context precision

**Then enters interactive mode:**
- Ask any question about the protocol in plain English
- Every answer includes a confidence score (0–100) and source page references
- Full session saved to `data/sample_output/session_timestamp.json`

**All outputs saved automatically:**

| File | Contents |
|---|---|
| `extraction_timestamp.json` | Structured criteria extracted from the protocol |
| `audit_report_timestamp.json` | Full audit findings from both agents |
| `risk_flags_timestamp.json` | Regulatory risk flags with suggested fixes |
| `evaluation_timestamp.json` | RAGAS-style scores across the test suite |
| `session_timestamp.json` | Interactive Q&A session log |

**Caching:**  
Phase 1 (embedding) and Phase 2 (extraction) are cached after the first run. Repeat runs load from disk, reducing startup time from ~3 minutes to ~10 seconds. Cache invalidates automatically if the PDF changes. To force a full re-run: `rm -rf data/cache/`

---

## Four Phases

### Phase 1 · Ingestion
Two complementary PDF loaders run in parallel. Standard loading handles prose sections quickly. Table-aware loading (Unstructured) targets pages with numerical criteria and preserves row/column relationships that plain text extraction destroys.

### Phase 2 · Structured Extraction
Extracts clinical criteria as schema-constrained JSON. A sanitizer strips markdown artifacts from LLM output before parsing. A retry prompt handles malformed responses. Results are written to disk as a versioned artifact.

### Phase 3 · Multi-Agent Audit
- **Agent 2 (Deterministic):** Checks numeric thresholds, standard terminology, and age-group coverage against a configurable rulebook. Fast, reproducible, no LLM required.
- **Agent 3 (LLM Second-Pass):** Re-reads source context to semantically verify the extraction. Catches subtle errors deterministic rules miss.

### Phase 3.5 · Ambiguity Detector
A Devil's Advocate agent that scans the protocol for language that could cause regulatory delay: ambiguous phrasing, missing specificity, or potential ICH/FDA guideline conflicts. Each finding includes a suggested fix.

### Phase 4 · Quantitative Evaluation
Runs a golden-dataset test suite scoring faithfulness and context precision.

> **Methodology note:** Faithfulness is measured via keyword-term matching (checking whether critical domain terms appear in the model's response). This serves as a lightweight hallucination proxy suitable for benchmarking pipeline changes. It is **not** equivalent to LLM-as-judge entailment scoring. In a production deployment, replace `run_ragas_style_eval()` with an LLM judge that checks whether each claim in the response is entailed by the retrieved context. Context precision measures whether the retriever surfaced at least one gold-standard page — a retrieval metric that is exact regardless of evaluation method.

**Results on the CAMPFIRE protocol:**

| Query | Faithfulness\* | Context Precision |
|---|---|---|
| Performance scores (§6.1) | 100% | 100% |
| Study rationale | 100% | 100% |
| End of Study definition | 100% | 100% |
| Organ transplant exclusion | 100% | 100% |
| **Average** | **100%** | **100%** |

*\*Faithfulness scored via keyword-term matching proxy. See methodology note above.*

---

## Prompt Design

All prompts live in `src/prompts.py`. Each follows the same structure:

```
[Role statement]       — who the model is playing
[Output format rules]  — schema, no markdown fences, null for missing values
[Fallback instruction] — what to say if the answer isn't in context
[Structured input]     — context and question in XML-like tags
[Chain-of-thought]     — "Think step by step"
```

| Template | Purpose |
|---|---|
| `CLINICAL_QA_PROMPT` | General protocol questions |
| `JSON_EXTRACTION_PROMPT` | Single-record structured extraction |
| `JSON_LIST_EXTRACTION_PROMPT` | Multi-record extraction (all age groups) |
| `AMBIGUITY_DETECTION_PROMPT` | Regulatory risk flagging |
| `CONFIDENCE_SCORED_PROMPT` | Answer + uncertainty quantification |
| `RETRY_EXTRACTION_PROMPT` | Fallback for malformed JSON responses |
| `AUDITOR_REVIEW_PROMPT` | LLM second-pass verification |

---

## Security

- API tokens loaded from `.env` via `python-dotenv` — never hardcoded
- `.env` and `data/cache/` are in `.gitignore` — never committed
- `.env.example` template provided for collaborators

---

## Dependencies

| Package | Role |
|---|---|
| `langchain` + `langchain-community` | RAG orchestration |
| `langchain-huggingface` | HF model integration |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | PubMedBERT embeddings |
| `rank-bm25` | Keyword retrieval |
| `pypdf` | Standard PDF loading |
| `unstructured[pdf]` | Table-aware PDF parsing |
| `opencv-python-headless` | Required by unstructured in headless environments |
| `python-dotenv` | Secure credential management |
| `pandas` | Tabular output and display |

---

## License
MIT