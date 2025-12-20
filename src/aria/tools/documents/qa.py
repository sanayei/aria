"""Document Question-Answering system using RAG (Retrieval Augmented Generation).

This module provides semantic question-answering capabilities over ingested documents
by combining vector search with LLM-based answer generation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from aria.llm.client import OllamaClient
from aria.llm.models import ChatMessage
from aria.memory.vectors import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class QASource(BaseModel):
    """Source information for a QA answer."""

    file_path: str
    file_name: str
    page_number: int | None = None
    chunk_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class QAResponse(BaseModel):
    """Response from a document Q&A query."""

    question: str
    answer: str
    sources: list[QASource]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    chunks_used: int
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str


class DocumentQA:
    """Question-answering system for ingested documents using RAG pattern.

    This class implements the Retrieval Augmented Generation (RAG) pattern:
    1. RETRIEVE: Search vector store for relevant document chunks
    2. AUGMENT: Build context from retrieved chunks
    3. GENERATE: Use LLM to generate answer with sources
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: OllamaClient,
        max_chunks: int = 5,
        min_relevance: float = 0.3,
    ):
        """Initialize DocumentQA system.

        Args:
            vector_store: Vector store containing document chunks
            llm_client: Ollama client for answer generation
            max_chunks: Maximum number of chunks to retrieve (default: 5)
            min_relevance: Minimum relevance score for chunks (default: 0.3)
        """
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._max_chunks = max_chunks
        self._min_relevance = min_relevance

    async def ask(
        self,
        question: str,
        document_filter: dict[str, Any] | None = None,
        include_sources: bool = True,
        max_chunks: int | None = None,
    ) -> QAResponse:
        """Ask a question about ingested documents.

        Uses RAG pattern:
        1. Searches vector store for relevant chunks
        2. Builds context from top chunks
        3. Generates answer using LLM with context
        4. Returns answer with source attribution

        Args:
            question: The question to answer
            document_filter: Optional metadata filter (e.g., {"file_type": ".pdf"})
            include_sources: Whether to include source citations (default: True)
            max_chunks: Override default max_chunks for this query

        Returns:
            QAResponse with answer, sources, and metadata

        Example:
            >>> qa = DocumentQA(vector_store, llm_client)
            >>> response = await qa.ask("What is the main topic of the paper?")
            >>> print(response.answer)
            >>> for source in response.sources:
            ...     print(f"From: {source.file_name} (page {source.page_number})")
        """
        logger.info(f"Answering question: {question}")

        # Step 1: RETRIEVE - Search for relevant chunks
        chunks_to_retrieve = max_chunks if max_chunks is not None else self._max_chunks
        search_results = await self._vector_store.search(
            query=question,
            limit=chunks_to_retrieve,
            metadata_filter=document_filter,
        )

        # Filter by relevance threshold
        relevant_results = [
            result for result in search_results if result.score >= self._min_relevance
        ]

        if not relevant_results:
            logger.warning(f"No relevant documents found for question: {question}")
            return QAResponse(
                question=question,
                answer="I couldn't find any relevant information in the ingested documents to answer this question.",
                sources=[],
                confidence=0.0,
                chunks_used=0,
                model_used=self._llm_client.model,
            )

        logger.info(
            f"Found {len(relevant_results)} relevant chunks "
            f"(scores: {[f'{r.score:.2f}' for r in relevant_results[:3]]})"
        )

        # Step 2: AUGMENT - Build context from chunks
        context = self._build_context(relevant_results)
        sources = self._build_sources(relevant_results)

        # Step 3: GENERATE - Generate answer using LLM
        answer, confidence = await self._generate_answer(
            question=question,
            context=context,
            include_sources=include_sources,
        )

        return QAResponse(
            question=question,
            answer=answer,
            sources=sources if include_sources else [],
            confidence=confidence,
            chunks_used=len(relevant_results),
            model_used=self._llm_client.model,
        )

    async def ask_about_document(
        self,
        question: str,
        file_path: str,
        include_sources: bool = True,
    ) -> QAResponse:
        """Ask a question about a specific document.

        Convenience method that filters to a single document.

        Args:
            question: The question to answer
            file_path: Path to the specific document
            include_sources: Whether to include source citations

        Returns:
            QAResponse with answer from the specified document
        """
        return await self.ask(
            question=question,
            document_filter={"file_path": file_path},
            include_sources=include_sources,
        )

    def _build_context(self, search_results: list[SearchResult]) -> str:
        """Build context string from search results.

        Formats chunks with source information for LLM consumption.
        """
        context_parts = []
        for i, result in enumerate(search_results, 1):
            file_name = result.metadata.get("file_name", "unknown")
            page_num = result.metadata.get("page_number")
            page_info = f" (page {page_num})" if page_num else ""

            context_parts.append(
                f"[Source {i}: {file_name}{page_info}]\n{result.content}\n"
            )

        return "\n".join(context_parts)

    def _build_sources(self, search_results: list[SearchResult]) -> list[QASource]:
        """Build list of QASource objects from search results."""
        sources = []
        for result in search_results:
            sources.append(
                QASource(
                    file_path=result.metadata.get("file_path", "unknown"),
                    file_name=result.metadata.get("file_name", "unknown"),
                    page_number=result.metadata.get("page_number"),
                    chunk_text=result.content,
                    relevance_score=result.score,
                )
            )
        return sources

    async def _generate_answer(
        self,
        question: str,
        context: str,
        include_sources: bool,
    ) -> tuple[str, float]:
        """Generate answer using LLM with retrieved context.

        Returns:
            Tuple of (answer, confidence_score)
        """
        # Build prompt with context
        source_instruction = (
            "Cite your sources using [Source N] notation when referencing information."
            if include_sources
            else ""
        )

        prompt = f"""You are a helpful assistant answering questions about documents.

Context from relevant documents:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information, say so clearly
- Be concise but complete in your answer
- {source_instruction}
- If you're uncertain, indicate your level of confidence

Answer:"""

        # Generate answer using LLM
        try:
            messages = [
                ChatMessage(role="user", content=prompt)
            ]
            response = await self._llm_client.chat(
                messages=messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=500,
            )

            answer = response.message.content.strip() if response.message.content else ""

            # Estimate confidence based on answer content
            # Simple heuristic: check for uncertainty phrases
            uncertainty_phrases = [
                "i don't know",
                "i'm not sure",
                "unclear",
                "not enough information",
                "cannot determine",
                "insufficient information",
            ]

            confidence = 0.8  # Default high confidence
            answer_lower = answer.lower()
            if any(phrase in answer_lower for phrase in uncertainty_phrases):
                confidence = 0.3
            elif "may" in answer_lower or "might" in answer_lower:
                confidence = 0.5
            elif "likely" in answer_lower or "probably" in answer_lower:
                confidence = 0.6

            return answer, confidence

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return (
                f"Error generating answer: {str(e)}",
                0.0,
            )

    async def get_relevant_context(
        self,
        query: str,
        max_chunks: int | None = None,
        document_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Get relevant document chunks without generating an answer.

        Useful for debugging or when you just need the context.

        Args:
            query: The query to search for
            max_chunks: Maximum chunks to retrieve
            document_filter: Optional metadata filter

        Returns:
            List of relevant SearchResult objects
        """
        chunks_to_retrieve = max_chunks if max_chunks is not None else self._max_chunks
        search_results = await self._vector_store.search(
            query=query,
            limit=chunks_to_retrieve,
            metadata_filter=document_filter,
        )

        return [
            result for result in search_results if result.score >= self._min_relevance
        ]
