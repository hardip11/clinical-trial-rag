"""
main.py — Clinical Trial RAG Pipeline: Interactive Mode

Usage:
    python main.py

Runs all 4 phases automatically, then enters an interactive Q&A loop
where you can ask questions about the protocol. All audit reports and
evaluation results are saved to data/sample_output/.

Caching:
    Phase 1 (embedding) and Phase 2 (extraction) results are cached to disk.
    On repeat runs, cached results are loaded instantly if the PDF hasn't changed.
    Cache is invalidated automatically when Protocol.pdf is modified.
    To force a full re-run: delete the data/cache/ folder.
"""

import json
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ── 0. Load environment ────────────────────────────────────────────────────
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("❌ HUGGINGFACEHUB_API_TOKEN not found.")
    print("   Create a .env file with your token. See .env.example.")
    sys.exit(1)

print("=" * 60)
print("  🏥 Clinical Trial RAG Pipeline")
print("=" * 60)
start_time = datetime.now()

# ── 1. Imports ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading models and libraries...")

sys.path.insert(0, "clinical-trial-rag")

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS

from src.ingestion import (
    load_pdf_standard,
    load_pdf_table_aware,
    format_elements_as_markdown,
    chunk_documents,
    build_vectorstore,
)
from src.retrieval import HybridRetriever
from src.evaluation import (
    RegulatoryAuditor,
    run_ragas_style_eval,
    save_extraction,
    sanitize_json_string,
)
from src.prompts import (
    CLINICAL_QA_PROMPT,
    JSON_LIST_EXTRACTION_PROMPT,
    AMBIGUITY_DETECTION_PROMPT,
    CONFIDENCE_SCORED_PROMPT,
    AUDITOR_REVIEW_PROMPT,
)

# ── 2. Configuration ───────────────────────────────────────────────────────
PDF_PATH = "data/Protocol.pdf"
OUTPUT_DIR = Path("data/sample_output")
CACHE_DIR = Path("data/cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INCLUSION_PAGES = list(range(15, 21))

# Cache file paths
CACHE_VECTORSTORE = CACHE_DIR / "vectorstore"
CACHE_CHUNKS = CACHE_DIR / "chunks.json"
CACHE_CONTEXT = CACHE_DIR / "markdown_context.txt"
CACHE_EXTRACTION = CACHE_DIR / "extracted_data.json"
CACHE_HASH = CACHE_DIR / "pdf_hash.txt"


# ── 3. Cache helpers ───────────────────────────────────────────────────────
def get_pdf_hash(pdf_path: str) -> str:
    """
    Compute MD5 hash of the PDF file.
    Used to detect if the PDF has changed since the last cache was built.
    """
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cache_is_valid(pdf_path: str) -> bool:
    """
    Check if all cache files exist and match the current PDF.
    Returns True if cache can be used, False if full re-run is needed.
    """
    required = [CACHE_VECTORSTORE, CACHE_CHUNKS, CACHE_CONTEXT, CACHE_EXTRACTION, CACHE_HASH]
    if not all(p.exists() for p in required):
        return False
    saved_hash = CACHE_HASH.read_text().strip()
    current_hash = get_pdf_hash(pdf_path)
    return saved_hash == current_hash


def save_cache(pdf_path: str, chunks, markdown_context: str, extracted_data: list, vectorstore):
    """Save all cacheable artifacts to disk."""
    # Save PDF hash
    CACHE_HASH.write_text(get_pdf_hash(pdf_path))

    # Save vectorstore
    vectorstore.save_local(str(CACHE_VECTORSTORE))

    # Save chunks as JSON (page_content + metadata only)
    chunks_data = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    CACHE_CHUNKS.write_text(json.dumps(chunks_data, indent=2))

    # Save markdown context
    CACHE_CONTEXT.write_text(markdown_context)

    # Save extraction
    CACHE_EXTRACTION.write_text(json.dumps(extracted_data, indent=2))

    print("💾 Cache saved to data/cache/")


def load_cache(embeddings):
    """Load all cacheable artifacts from disk."""
    from langchain_core.documents import Document

    vectorstore = FAISS.load_local(
        str(CACHE_VECTORSTORE),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    chunks_data = json.loads(CACHE_CHUNKS.read_text())
    chunks = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks_data]
    markdown_context = CACHE_CONTEXT.read_text()
    extracted_data = json.loads(CACHE_EXTRACTION.read_text())

    return chunks, markdown_context, extracted_data, vectorstore


# ── 4. Initialize models ───────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
    max_new_tokens=512,
    temperature=0.1,
    timeout=300,
)
llm = ChatHuggingFace(llm=llm_endpoint)
print("✅ Models loaded.")

# ── Check PDF exists ───────────────────────────────────────────────────────
if not Path(PDF_PATH).exists():
    print(f"❌ Protocol PDF not found at '{PDF_PATH}'.")
    print("   Place your PDF in the project root and name it 'Protocol.pdf'.")
    sys.exit(1)

# ── Phase 1: Ingestion (with caching) ─────────────────────────────────────
print("\n[2/5] Phase 1: Ingesting protocol PDF...")

if cache_is_valid(PDF_PATH):
    print("⚡ Cache hit — loading from disk (skipping embedding)...")
    chunks, markdown_context, extracted_data, vectorstore = load_cache(embeddings)
    print(f"   Loaded {len(chunks)} chunks and vectorstore from cache.")
    cache_hit = True
else:
    if CACHE_HASH.exists():
        print("🔄 PDF has changed — rebuilding cache...")
    else:
        print("🆕 No cache found — running full ingestion...")
    cache_hit = False

    docs = load_pdf_standard(PDF_PATH)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, embeddings)

    table_elements = load_pdf_table_aware(PDF_PATH, target_pages=INCLUSION_PAGES)
    markdown_context = format_elements_as_markdown(table_elements)

print("✅ Phase 1 complete.")

retriever = HybridRetriever(chunks, vectorstore, k=3)

# ── Phase 2: Structured Extraction (with caching) ─────────────────────────
print("\n[3/5] Phase 2: Extracting structured criteria...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if cache_hit:
    print("⚡ Cache hit — using cached extraction.")
else:
    extraction_target = "ALL Performance Score requirements (Lansky and Karnofsky scores)"
    prompt_text = JSON_LIST_EXTRACTION_PROMPT.format(
        context=markdown_context,
        extraction_target=extraction_target,
    )

    raw_output = llm.invoke(prompt_text).content
    clean_json_str = sanitize_json_string(raw_output)

    if clean_json_str:
        extracted_data = json.loads(clean_json_str)
    else:
        print("⚠️  First extraction attempt failed. Retrying with stricter prompt...")
        retry_prompt = f"""
Output ONLY a JSON array. Nothing else. No text before or after.

Schema: [{{"age_group": "<16" or ">=16", "metric": string, "minimum_score": integer, "section": string}}]

Extract all performance score requirements from this context:
{markdown_context[:3000]}
"""
        raw_retry = llm.invoke(retry_prompt).content
        clean_json_str = sanitize_json_string(raw_retry)
        extracted_data = json.loads(clean_json_str) if clean_json_str else []

    # Save cache after both phases complete
    save_cache(PDF_PATH, chunks, markdown_context, extracted_data, vectorstore)

# Always save a timestamped extraction artifact
extraction_path = OUTPUT_DIR / f"extraction_{timestamp}.json"
save_extraction(extracted_data, extraction_path)

print("\n--- EXTRACTED CRITERIA ---")
for record in extracted_data:
    print(f"  {record.get('age_group'):6} | {record.get('metric'):30} | Score ≥ {record.get('minimum_score')} | §{record.get('section')}")

print("✅ Phase 2 complete.")

# ── Phase 3: Multi-Agent Audit ─────────────────────────────────────────────
print("\n[4/5] Phase 3: Running multi-agent audit...")

# Agent 2: Deterministic auditor
auditor = RegulatoryAuditor()
report = auditor.run(extracted_data)
report.print()

# Agent 3: LLM second-pass
print("\nRunning LLM second-pass audit...")
auditor_prompt = AUDITOR_REVIEW_PROMPT.format(
    prior_extraction=json.dumps(extracted_data, indent=2),
    context=markdown_context,
)
raw_audit = llm.invoke(auditor_prompt).content
clean_audit = sanitize_json_string(raw_audit)

llm_audit_result = {}
if clean_audit:
    llm_audit_result = json.loads(clean_audit)
    if isinstance(llm_audit_result, list):
        llm_audit_result = llm_audit_result[0] if llm_audit_result else {}
    verdict = llm_audit_result.get("recommendation", "unknown").upper()
    verified = llm_audit_result.get("verified", False)
    print(f"  LLM Verdict   : {verdict}")
    print(f"  Verified      : {'✅' if verified else '❌'}")
    issues = llm_audit_result.get("issues", [])
    if issues:
        for issue in issues:
            sev = issue.get("severity", "").upper()
            icon = "❌" if sev == "CRITICAL" else "⚠️"
            print(f"  {icon} [{sev}] {issue.get('field')}: {issue.get('concern', '')}")
    else:
        print("  ✅ No issues found by LLM auditor.")

# Save audit report
audit_report = {
    "timestamp": timestamp,
    "deterministic_audit": {
        "is_compliant": report.is_compliant,
        "recommendation": report.recommendation,
        "findings": [
            {"status": f.status, "metric": f.metric, "score": f.score, "message": f.message}
            for f in report.findings
        ],
    },
    "llm_audit": llm_audit_result,
}
audit_path = OUTPUT_DIR / f"audit_report_{timestamp}.json"
with open(audit_path, "w") as f:
    json.dump(audit_report, f, indent=4)
print(f"💾 Audit report saved to: {audit_path}")
print("✅ Phase 3 complete.")

# ── Phase 3.5: Ambiguity Detection ────────────────────────────────────────
print("\n[4.5/5] Phase 3.5: Scanning for regulatory risks...")

broad_docs = retriever.retrieve("patient eligibility criteria inclusion exclusion", verbose=False)
broad_context = "\n\n".join(doc.page_content for doc in broad_docs)
ambiguity_prompt = AMBIGUITY_DETECTION_PROMPT.format(context=broad_context)

raw_flags = llm.invoke(ambiguity_prompt).content
clean_flags = sanitize_json_string(raw_flags)

if clean_flags:
    flags = json.loads(clean_flags)
    print("\n--- REGULATORY RISK FLAGS ---")
    for i, flag in enumerate(flags, 1):
        issue_type = flag.get("issue_type", "unknown").upper()
        print(f"\n  🔍 Issue {i} [{issue_type}]")
        print(f"     Quote   : \"{flag.get('quote', 'N/A')}\"")
        print(f"     Concern : {flag.get('concern', 'N/A')}")
        print(f"     Fix     : {flag.get('suggested_fix', 'N/A')}")

    flags_path = OUTPUT_DIR / f"risk_flags_{timestamp}.json"
    with open(flags_path, "w") as f:
        json.dump(flags, f, indent=4)
    print(f"\n💾 Risk flags saved to: {flags_path}")

# ── Phase 4: Evaluation ────────────────────────────────────────────────────
print("\n[5/5] Phase 4: Running evaluation suite...")

GOLDEN_DATASET = [
    {
        "query": "What are the specific Lansky and Karnofsky score requirements in Section 6.1?",
        "gold_pages": [15, 16, "15", "16"],
        "faithfulness_terms": ["50", "Lansky", "Karnofsky"],
    },
    {
        
        "query": "What is the primary rationale for using a master protocol in the CAMPFIRE study?",
        

        "gold_pages": [10, 11, "10", "11", 12, "12"],

        "faithfulness_terms": ["pediatric", "master", "protocol"],
    },
    {
        

        "query": "How is End of Study defined for an individual investigation?",
        "gold_pages": [14, 15, "14", "15"],

        "faithfulness_terms": ["end", "study", "investigation"],
    },
    {
        "query": "What are the exclusion criteria for patients with prior organ transplants?",
        "gold_pages": [19, 20, "19", "20"],
        "faithfulness_terms": ["transplant", "organ", "exclusion"],
    },
]

eval_results = []
for item in GOLDEN_DATASET:
    query = item["query"]
    context_docs = retriever.retrieve(query, verbose=False)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    qa_prompt_text = CLINICAL_QA_PROMPT.format(context=context_text, question=query)
    response = llm.invoke(qa_prompt_text).content

    result = run_ragas_style_eval(
        query=query,
        response=response,
        context_docs=context_docs,
        gold_pages=item["gold_pages"],
        faithfulness_terms=item["faithfulness_terms"],
    )
    result.print()
    eval_results.append({
        "query": result.query,
        "faithfulness": result.faithfulness,
        "context_precision": result.context_precision,
        "retrieved_pages": result.retrieved_pages,
        "gold_pages": result.gold_pages,
    })

avg_faith = sum(r["faithfulness"] for r in eval_results) / len(eval_results)
avg_prec = sum(r["context_precision"] for r in eval_results) / len(eval_results)

print(f"\n=== PIPELINE SCORECARD ===")
print(f"  Average Faithfulness      : {avg_faith * 100:.1f}%")
print(f"  Average Context Precision : {avg_prec * 100:.1f}%")

if avg_faith >= 0.8 and avg_prec >= 0.75:
    print("  🚀 REGULATORY GRADE — pipeline cleared for deployment.")
elif avg_faith >= 0.8:
    print("  ⚠️  Answers accurate but source traceability needs improvement.")
else:
    print("  ❌ Hallucination risk detected. Review retrieval and prompts.")

eval_path = OUTPUT_DIR / f"evaluation_{timestamp}.json"
with open(eval_path, "w") as f:
    json.dump({
        "timestamp": timestamp,
        "results": eval_results,
        "scorecard": {
            "avg_faithfulness": avg_faith,
            "avg_context_precision": avg_prec,
        }
    }, f, indent=4)
print(f"💾 Evaluation results saved to: {eval_path}")
print("✅ Phase 4 complete.")

# ── Interactive Q&A ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  🤖 Interactive Protocol Q&A")
print("  Ask any question about the protocol.")
print("  Type 'quit' to exit.")
print("=" * 60)

session_log = []


while True:
    print()
    query = input("Your question: ").strip()

    if query.lower() in ("quit", "exit", "q"):
        print("\nSession ended.")
        break

    if not query:
        continue

    try:
        # Retrieve context
        context_docs = retriever.retrieve(query, verbose=False)
        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        # Try confidence-scored answer first
        conf_prompt = CONFIDENCE_SCORED_PROMPT.format(
            context=context_text,
            question=query,
        )
        raw = llm.invoke(conf_prompt).content
        clean = sanitize_json_string(raw)

        confidence = "?"
        answer = "N/A"
        reasoning = "N/A"
        pages = []

        if clean:
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            if isinstance(parsed, dict):
                confidence = parsed.get("confidence", "?")
                answer = parsed.get("answer", "N/A")
                reasoning = parsed.get("reasoning", "N/A")
                pages = parsed.get("source_pages", [])

        # Fallback to plain QA if parsing failed
        if answer == "N/A":
            qa_prompt_text = CLINICAL_QA_PROMPT.format(
                context=context_text,
                question=query,
            )
            answer = llm.invoke(qa_prompt_text).content
            confidence = "N/A"

        # Display result
        if str(confidence).isdigit():
            icon = "✅" if int(confidence) >= 80 else "⚠️" if int(confidence) >= 50 else "❌"
            print(f"\n  Answer     : {answer}")
            print(f"  Confidence : {icon} {confidence}/100")
            print(f"  Reasoning  : {reasoning}")
            if pages:
                print(f"  Source pages: {pages}")
        else:
            print(f"\n  Answer: {answer}")

        session_log.append({
            "query": query,
            "answer": answer,
            "confidence": confidence,
        })

    except Exception as e:
        print(f"\n  ⚠️  Error: {e}")
        print("  Try rephrasing your question.")
        continue

# Save session log
if session_log:
    session_path = OUTPUT_DIR / f"session_{timestamp}.json"
    with open(session_path, "w") as f:
        json.dump(session_log, f, indent=4)
    print(f"💾 Session log saved to: {session_path}")

elapsed = datetime.now() - start_time
minutes, seconds = divmod(elapsed.seconds, 60)
print(f"\nAll outputs saved to data/sample_output/")
print(f"Total runtime: {minutes}m {seconds}s")
print("Done. ✅")