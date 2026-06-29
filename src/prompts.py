"""
prompts.py — All prompt templates for the Clinical Trial RAG Pipeline.

Design principles:
- Every prompt has a role (who the model is playing)
- Every prompt has explicit output format instructions
- Every prompt has a fallback instruction ("if not in context, say so")
- JSON prompts enforce strict schema and forbid markdown fences

Templates (7 total):
    1. CLINICAL_QA_PROMPT            — general protocol questions
    2. JSON_EXTRACTION_PROMPT        — single-record structured extraction
    3. JSON_LIST_EXTRACTION_PROMPT   — multi-record extraction
    4. AMBIGUITY_DETECTION_PROMPT    — regulatory risk flagging
    5. CONFIDENCE_SCORED_PROMPT      — answer + uncertainty quantification
    6. RETRY_EXTRACTION_PROMPT       — fallback for malformed JSON responses
    7. AUDITOR_REVIEW_PROMPT         — LLM second-pass verification
"""

from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# 1. Core QA Prompt
#    Used for general protocol questions via the hybrid retriever.
# ---------------------------------------------------------------------------

CLINICAL_QA_PROMPT = PromptTemplate.from_template("""
You are a Regulatory Affairs AI assistant specializing in clinical trial protocol analysis.

Your responses must be:
- Grounded ONLY in the provided context (cite page numbers when possible)
- Precise with numerical values (scores, timeframes, dosages)
- Structured as bullet points for multi-part answers
- Honest when information is absent: respond "Not specified in the provided context"

<context>
{context}
</context>

<question>
{question}
</question>

Think step by step:
1. Identify the relevant section(s) in the context above
2. Extract exact values or criteria, preserving any age/group qualifiers
3. Note page numbers if available in the metadata
4. If you found a value in step 2, your final answer MUST reflect that value. Do NOT say "not specified" if you already extracted the answer above.                                                  
                                                

Answer:""")


# ---------------------------------------------------------------------------
# 2. Structured JSON Extraction Prompt (single object)
#    Used for targeted extraction of one specific data type.
# ---------------------------------------------------------------------------

JSON_EXTRACTION_PROMPT = PromptTemplate.from_template("""
You are a clinical data parser. Your only job is to extract structured data.

RULES:
- Output ONLY valid JSON. No markdown backticks. No explanation before or after.
- Use null for any field you cannot find — never invent values.
- age_group must be one of: "<16", ">=16", "all"

SCHEMA (single object):
{{
  "age_group": string,
  "metric": string,
  "minimum_score": integer or null,
  "section": string or null
}}

Context:
{context}

Extract: {extraction_target}
""")


# ---------------------------------------------------------------------------
# 3. Structured JSON List Extraction Prompt (multiple objects)
#    Used when a criterion appears more than once (e.g., both age groups).
# ---------------------------------------------------------------------------

JSON_LIST_EXTRACTION_PROMPT = PromptTemplate.from_template("""
You are a clinical data parser. Your only job is to extract a complete list of structured records.

RULES:
- Output ONLY a valid JSON array. No markdown. No explanation.
- Capture ALL instances — do not stop after finding the first match.
- Use null for any missing field. Do not invent values.
- age_group must be one of: "<16", ">=16", "all"

SCHEMA (list of objects):
[
  {{
    "age_group": string,
    "metric": string,
    "minimum_score": integer or null,
    "section": string or null
  }}
]

Context:
{context}

Extract all instances of: {extraction_target}
""")


# ---------------------------------------------------------------------------
# 4. Ambiguity / Conflict Detection Prompt
#    Creative addition: surfaces regulatory risks in the protocol text.
# ---------------------------------------------------------------------------

AMBIGUITY_DETECTION_PROMPT = PromptTemplate.from_template("""
You are a senior Regulatory Affairs specialist performing a Devil's Advocate review.

Your task: identify statements in the protocol that could cause regulatory delay or audit findings.
Look for:
1. Ambiguous phrasing (open to two or more interpretations)
2. Missing specificity (timeframes without units, criteria without thresholds)
3. Potential conflicts with common ICH E6 / FDA 21 CFR Part 312 requirements

RULES:
- Output ONLY a valid JSON array. No markdown fences. No preamble.
- Limit to the 3 most significant issues.
- Be specific: quote the exact phrase that is problematic.

SCHEMA:
[
  {{
    "issue_type": "ambiguity" | "missing_specificity" | "guideline_conflict",
    "quote": string,
    "concern": string,
    "suggested_fix": string
  }}
]

Context:
{context}
""")


# ---------------------------------------------------------------------------
# 5. Confidence-Scored Answer Prompt
#    Wraps any answer in a self-assessed confidence rating.
#    Critical for clinical AI — surfacing uncertainty is a safety feature.
# ---------------------------------------------------------------------------

CONFIDENCE_SCORED_PROMPT = PromptTemplate.from_template("""
You are a Regulatory Affairs AI assistant. Answer the question using ONLY the context provided.

After answering, rate your confidence from 0–100:
- 90–100: The answer is stated explicitly and verbatim in the context
- 60–89: The answer is strongly implied but requires one inferential step
- 30–59: The answer is partially supported; key details may be missing
- 0–29: The context does not clearly support the answer

RULES:
- Output ONLY a valid JSON object. No markdown. No preamble.

SCHEMA:
{{
  "answer": string,
  "confidence": integer,
  "reasoning": string,
  "source_pages": list of integers or null
}}

Context:
{context}

Question: {question}
""")


# ---------------------------------------------------------------------------
# 6. Audit / Regulatory Rulebook Prompt
#    Used by the Auditor Agent to re-examine a prior extraction.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7. Retry Extraction Prompt
#    Fallback used when JSON_LIST_EXTRACTION_PROMPT returns malformed output.
#    Stricter wording, truncated context, and an explicit "nothing else" rule
#    to recover from LLMs that add prose or markdown fences on first pass.
# ---------------------------------------------------------------------------

RETRY_EXTRACTION_PROMPT = PromptTemplate.from_template("""Output ONLY a valid JSON array. Nothing else. No text before or after. No markdown.

SCHEMA:
[
  {{
    "age_group": "<16" or ">=16",
    "metric": string,
    "minimum_score": integer,
    "section": string
  }}
]

Extract all performance score requirements from this context:
{context}
""")


# ---------------------------------------------------------------------------
# 7. Audit / Regulatory Rulebook Prompt
#    Used by the Auditor Agent to re-examine a prior extraction.
# ---------------------------------------------------------------------------

AUDITOR_REVIEW_PROMPT = PromptTemplate.from_template("""
You are a Regulatory Compliance Auditor performing a second-pass review.

A previous AI extraction produced this output:
{prior_extraction}

Your task: verify this output against the source context below.
Check for:
- Missing records (are all age groups / populations covered?)
- Numeric accuracy (do the extracted scores match the context exactly?)
- Terminology consistency (do metric names match standard clinical terminology?)

RULES:
- Output ONLY a valid JSON object. No markdown. No preamble.

SCHEMA:
{{
  "verified": boolean,
  "issues": [
    {{
      "field": string,
      "expected": string,
      "found": string,
      "severity": "critical" | "warning" | "info"
    }}
  ],
  "recommendation": "approve" | "revise" | "escalate"
}}

Source context:
{context}
""")