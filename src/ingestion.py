"""
ingestion.py — PDF loading, table-aware parsing, and vector store creation.

Handles two ingestion strategies:
1. Standard (RecursiveCharacterTextSplitter) — fast, good for prose
2. Table-aware (UnstructuredPDFLoader) — slower, preserves table structure
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def build_embeddings(model_name: str = "NeuML/pubmedbert-base-embeddings") -> HuggingFaceEmbeddings:
    """
    Load a domain-specific sentence embedding model.

    Default: PubMedBERT — trained on biomedical literature, outperforms
    general-purpose models on clinical terminology matching.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    print(f"Loading embeddings model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


# ---------------------------------------------------------------------------
# PDF Loading
# ---------------------------------------------------------------------------

def load_pdf_standard(pdf_path: str | Path) -> list[Document]:
    """
    Load a PDF with PyPDF (text-layer extraction).

    Fast. Works well for prose-heavy protocols.
    Tables may be parsed as flat strings — use load_pdf_table_aware() if
    structured tables are critical to your queries.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects, one per page.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Protocol PDF not found: {path}. "
                                f"Place your PDF in the project root and update the path.")
    print(f"Loading PDF (standard): {path.name}")
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    print(f"  → Loaded {len(docs)} pages")
    return docs


def load_pdf_table_aware(
    pdf_path: str | Path,
    target_pages: Optional[list[int]] = None
) -> list[Document]:
    """
    Load a PDF using Unstructured, which identifies and preserves table structure.

    Slower than standard loading (~2–5× longer), but critical for protocols
    where numerical criteria appear in tables (e.g., inclusion/exclusion scores).

    Args:
        pdf_path: Path to the PDF file.
        target_pages: If provided, filter elements to only these page numbers
                      (1-indexed). Useful for targeting specific sections.

    Returns:
        List of Document objects with category metadata (Table / NarrativeText / etc.)
    """
    path = Path(pdf_path)
    print(f"Loading PDF (table-aware): {path.name}")
    loader = UnstructuredPDFLoader(str(path), strategy="fast", mode="elements")
   
    try:
        elements = loader.load()
    except Exception as e:
        print(f"⚠️  Table-aware loader failed: {e}")
        print("   Falling back to standard loader for table pages...")
        fallback_loader = PyPDFLoader(str(path))
        return fallback_loader.load()
    

    if target_pages:
        elements = [
            el for el in elements
            if el.metadata.get("page_number") in target_pages
        ]
        print(f"  → Filtered to pages {target_pages}: {len(elements)} elements")
    else:
        print(f"  → Loaded {len(elements)} elements")

    return elements


def format_elements_as_markdown(elements: list[Document]) -> str:
    """
    Convert Unstructured elements into a Markdown string that LLMs can parse.

    Tables are wrapped in ### [TABLE] headers so the model knows to treat
    them as structured data rather than prose.

    Args:
        elements: Raw elements from UnstructuredPDFLoader.

    Returns:
        Formatted Markdown string.
    """
    parts = []
    if not elements:
        print("⚠️  No elements found for target pages. Check page numbers.")
        return ""

    for el in elements:
        category = el.metadata.get("category", "NarrativeText")
        if category == "Table":
            parts.append(f"\n### [TABLE — Page {el.metadata.get('page_number', '?')}]\n{el.page_content}\n")
        else:
            parts.append(f"\n{el.page_content}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split documents into overlapping chunks for retrieval.

    Chunk size of 1200 characters balances context richness (enough text
    to answer a question) with retrieval precision (small enough to rank well).
    Overlap of 200 prevents criteria that span chunk boundaries from being lost.

    Args:
        docs: Raw documents from a loader.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks.

    Returns:
        List of chunk Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

def build_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = 10,
) -> FAISS:
    """
    Embed chunks and store them in a FAISS index.

    Uses batched processing with a progress bar so long PDFs don't appear
    to hang. Each batch of 10 chunks is embedded and added to the index.

    Args:
        chunks: Text chunks to embed.
        embeddings: Embedding model instance.
        batch_size: Number of chunks to embed per API call.

    Returns:
        Populated FAISS vectorstore.
    """
    print(f"Embedding {len(chunks)} chunks into PubMedBERT vectors...")
    vectorstore = None

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Progress"):
        batch = chunks[i: i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    print("✅ Vector store ready.")
    return vectorstore