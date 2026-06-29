"""
evaluation.py — Quantitative RAG evaluation and multi-agent audit logic.

Two complementary validation layers:

1. RegulatoryAuditor: deterministic rule-checking against a safety rulebook.
   Fast, transparent, reproducible. Catches numeric threshold violations and
   non-standard terminology. Runs on every extraction.

2. run_ragas_style_eval: lightweight RAGAS-inspired scoring.
   Measures faithfulness (no hallucinations) and context precision (did the
   retriever find the right pages?). Used for benchmarking pipeline changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuditFinding:
    """A single finding from the regulatory auditor."""
    status: str          # "PASS", "FAIL", "WARNING", "ERROR"
    metric: str
    score: Optional[int]
    message: str


@dataclass
class AuditReport:
    """Full audit report returned by RegulatoryAuditor.run()."""
    is_compliant: bool
    findings: list[AuditFinding] = field(default_factory=list)
    recommendation: str = "approve"

    def print(self) -> None:
        """Pretty-print the audit report to stdout."""
        print("\n--- MULTI-AGENT AUDIT REPORT ---")
        icons = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️", "ERROR": "❌"}
        for f in self.findings:
            print(f"{icons.get(f.status, '?')} {f.status}: {f.message}")
        verdict = "🚀 APPROVED: Ready for Regulatory Submission." if self.is_compliant \
            else "🔄 ACTION REQUIRED: See findings above."
        print(f"\n{verdict}")


@dataclass
class EvalResult:
    """Result of a single RAGAS-style evaluation run."""
    query: str
    faithfulness: float       # 0.0 – 1.0
    context_precision: float  # 0.0 – 1.0
    retrieved_pages: list[str]
    gold_pages: list[str]

    def print(self) -> None:
        print(f"\n--- EVALUATION: {self.query[:55]}... ---")
        print(f"  📊 Faithfulness      : {self.faithfulness * 100:.0f}%")
        print(f"  📊 Context Precision : {self.context_precision * 100:.0f}%")
        print(f"  Retrieved pages: {self.retrieved_pages}")
        print(f"  Gold pages:      {self.gold_pages}")
        if self.faithfulness == 1.0 and self.context_precision == 1.0:
            print("  ✅ REGULATORY GRADE — ready for deployment.")
        elif self.faithfulness == 1.0:
            print("  ⚠️  Correct answer but source traceability unclear.")
        else:
            print("  ❌ Hallucination risk detected — review retrieval.")


# ---------------------------------------------------------------------------
# Deterministic Auditor (Agent 2)
# ---------------------------------------------------------------------------

class RegulatoryAuditor:
    """
    Deterministic second-pass validation agent.

    Simulates the role of a Regulatory Affairs Specialist who reviews
    AI-extracted data before it enters a submission workflow.

    The rulebook is intentionally separate from the extraction logic so
    it can be updated when FDA/ICH guidelines change without touching
    the RAG pipeline.
    """

    DEFAULT_RULEBOOK = {
        "min_safe_score": 50,
        "required_metrics": ["Lansky", "Karnofsky"],
        "required_age_groups": ["<16", ">=16"],
    }

    def __init__(self, rulebook: Optional[dict] = None) -> None:
        self.rulebook = rulebook or self.DEFAULT_RULEBOOK

    def run(self, extracted_json: list[dict]) -> AuditReport:
        """
        Validate extracted JSON records against the rulebook.

        Args:
            extracted_json: List of dicts with keys:
                            age_group, metric, minimum_score, section

        Returns:
            AuditReport with all findings and a compliance decision.
        """
        findings: list[AuditFinding] = []
        is_compliant = True

        # --- Check 1: Numeric threshold ---
        for entry in extracted_json:
            raw_val = str(entry.get("minimum_score", "0") or "0")
            match = re.search(r"\d+", raw_val)

            if not match:
                findings.append(AuditFinding(
                    status="ERROR",
                    metric=entry.get("metric", "unknown"),
                    score=None,
                    message=f"Could not parse numeric score from '{raw_val}'"
                ))
                is_compliant = False
                continue

            score = int(match.group())
            metric = entry.get("metric", "unknown")

            if score < self.rulebook["min_safe_score"]:
                findings.append(AuditFinding(
                    status="FAIL",
                    metric=metric,
                    score=score,
                    message=f"{metric} score ({score}) is BELOW the safety threshold "
                            f"(≥{self.rulebook['min_safe_score']})"
                ))
                is_compliant = False
            else:
                findings.append(AuditFinding(
                    status="PASS",
                    metric=metric,
                    score=score,
                    message=f"{metric} ({score}) meets the safety threshold."
                ))

            # --- Check 2: Standard terminology ---
            if not any(m in metric for m in self.rulebook["required_metrics"]):
                findings.append(AuditFinding(
                    status="WARNING",
                    metric=metric,
                    score=score,
                    message=f"Metric '{metric}' deviates from standard terminology "
                            f"({', '.join(self.rulebook['required_metrics'])}). "
                            f"Flag for medical writer review."
                ))

        # --- Check 3: Coverage — both age groups present ---
        found_groups = {e.get("age_group", "") for e in extracted_json}
        for required_group in self.rulebook["required_age_groups"]:
            if required_group not in found_groups:
                findings.append(AuditFinding(
                    status="FAIL",
                    metric="age_group_coverage",
                    score=None,
                    message=f"Missing criteria for age group '{required_group}'. "
                            f"Incomplete extraction — do not submit."
                ))
                is_compliant = False

        recommendation = "approve" if is_compliant else \
            ("escalate" if any(f.status == "ERROR" for f in findings) else "revise")

        return AuditReport(
            is_compliant=is_compliant,
            findings=findings,
            recommendation=recommendation,
        )


# ---------------------------------------------------------------------------
# RAGAS-style Evaluation
# ---------------------------------------------------------------------------

def run_ragas_style_eval(
    query: str,
    response: str,
    context_docs: list[Document],
    gold_pages: list[int | str],
    faithfulness_terms: Optional[list[str]] = None,
) -> EvalResult:
    """
    Lightweight evaluation inspired by the RAGAS framework.

    Measures two KPIs without requiring a separate evaluation LLM:

    - Faithfulness: checks whether critical domain terms appear in the response.
      Proxy for hallucination resistance. In a production system, replace with
      an LLM-as-judge call using an entailment prompt.

    - Context Precision: checks whether the retriever surfaced at least one
      gold-standard page. Zero means the AI answered from memory, not from
      the document — a regulatory red flag.

    Args:
        query: The original question.
        response: The LLM's generated answer.
        context_docs: Documents returned by the retriever.
        gold_pages: Page numbers expected to contain the answer.
        faithfulness_terms: Domain terms that must appear in the response.
                            Defaults to clinical performance score terms.

    Returns:
        EvalResult with scores and retrieved page metadata.
    """
    if faithfulness_terms is None:
        faithfulness_terms = ["50", "Lansky", "Karnofsky"]

    # Faithfulness: term presence as a hallucination proxy
    found = [t for t in faithfulness_terms if t.lower() in response.lower()]
    faithfulness = len(found) / len(faithfulness_terms)

    # Context Precision: did the retriever find any gold page?
    retrieved_pages = []
    for doc in context_docs:
        # Normalize: Unstructured uses 'page_number', PyPDF uses 'page'
        page = doc.metadata.get("page_number") or doc.metadata.get("page")
        if page is not None:
            retrieved_pages.append(str(page))

    gold_str = [str(g) for g in gold_pages]
    precision = 1.0 if any(p in retrieved_pages for p in gold_str) else 0.0

    return EvalResult(
        query=query,
        faithfulness=faithfulness,
        context_precision=precision,
        retrieved_pages=retrieved_pages,
        gold_pages=gold_str,
    )


# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------

def save_extraction(data: list[dict], output_path: str | Path) -> None:
    """
    Sanitize and persist an extraction result as a JSON file.

    Strips any markdown fences the LLM may have emitted, validates that the
    result is a proper list, then writes it with 4-space indentation.

    Args:
        data: Extracted records (already parsed from LLM output).
        output_path: Destination path for the JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"💾 Saved extraction to: {path}")


def sanitize_json_string(raw: str) -> Optional[str]:
    """
    Extract the first valid JSON array or object from an LLM response string.

    LLMs occasionally prefix JSON with prose or wrap it in markdown fences.
    This function strips those artifacts so json.loads() can parse cleanly.

    Args:
        raw: Raw LLM response text.

    Returns:
        Cleaned JSON string, or None if no valid JSON structure was found.
    """
    # Try array first, then object
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(0)
    return None