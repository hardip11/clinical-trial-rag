"""
retrieval.py — Hybrid BM25 + semantic retrieval.

Why hybrid?
- Semantic (vector) search: excellent at intent matching. Finds "health status"
  when the user asks about "performance score."
- BM25 (keyword) search: exact string matching. Reliably finds "Section 6.1"
  or drug identifiers like "LY900023" that may be rare in the embedding space.

Combining both minimizes the failure modes of each approach.
"""

from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class HybridRetriever:
    """
    Merges BM25 keyword results and FAISS semantic results, deduplicating
    by page content so no chunk is passed to the LLM twice.

    Usage:
        retriever = HybridRetriever(chunks, vectorstore, k=3)
        docs = retriever.retrieve("What is the Lansky score requirement?")
    """

    def __init__(
        self,
        chunks: list[Document],
        vectorstore: FAISS,
        k: int = 3,
    ) -> None:
        """
        Args:
            chunks: The same chunks used to build the vectorstore.
            vectorstore: Populated FAISS index.
            k: Number of results to fetch from each retriever before merging.
        """
        self.k = k
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = k
        self.semantic = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"HybridRetriever initialized (k={k} per engine, up to {k * 2} unique results)")

    def retrieve(self, query: str, verbose: bool = True) -> list[Document]:
        """
        Run both retrievers and return deduplicated results.

        Args:
            query: The user's question.
            verbose: If True, prints retrieval stats.

        Returns:
            Deduplicated list of Documents, keyword results first.
        """
        keyword_docs = self.bm25.invoke(query)
        semantic_docs = self.semantic.invoke(query)

        seen: set[str] = set()
        unique: list[Document] = []

        for doc in keyword_docs + semantic_docs:
            content_key = doc.page_content.strip()
            if content_key not in seen:
                unique.append(doc)
                seen.add(content_key)

        if verbose:
            print(f"  BM25: {len(keyword_docs)} | Semantic: {len(semantic_docs)} | "
                  f"Unique merged: {len(unique)}")

        return unique

    def retrieve_as_text(self, query: str) -> str:
        """
        Retrieve and join all chunks into a single context string for prompting.

        Args:
            query: The user's question.

        Returns:
            Joined context string.
        """
        docs = self.retrieve(query)
        return "\n\n".join(doc.page_content for doc in docs)