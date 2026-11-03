
import hashlib
import logging
import re

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker


logger = logging.getLogger(__name__)


class SemanticChunkingService:
    """
    Convert page-level documents into semantic chunks.

    Embeddings are injected to keep the service independent
    of the embedding provider.
    """

    def __init__(self, embeddings):
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

    def chunk_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:

        if not documents:
            raise ValueError("No documents supplied for chunking")

        all_chunks = []

        for document in documents:

            text = document.page_content.strip()

            if not text:
                logger.warning(
                    "Skipping empty page: %s",
                    document.metadata.get("page_number")
                )
                continue

            metadata = document.metadata

            # Preserve financial tables as a complete page
            # during the initial implementation.
            has_tables = metadata.get("table_count", 0) > 0

            # Avoid semantic splitting when a page contains
            # too little sentence-level information.
            sentences = [
                sentence for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    text
                )
                if sentence.strip()
            ]

            if has_tables or len(sentences) < 3:
                page_chunks = [
                    Document(
                        page_content=text,
                        metadata=metadata.copy()
                    )
                ]

            else:
                page_chunks = self.splitter.create_documents(
                    texts=[text],
                    metadatas=[metadata.copy()]
                )

            for chunk_index, chunk in enumerate(page_chunks):

                chunk_text = chunk.page_content.strip()

                if not chunk_text:
                    continue

                content_hash = hashlib.sha256(
                    chunk_text.encode("utf-8")
                ).hexdigest()

                chunk.metadata.update({
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                })

                all_chunks.append(chunk)

        if not all_chunks:
            raise ValueError(
                "Semantic chunking produced no valid chunks"
            )

        logger.info(
            "Generated %d semantic chunks from %d pages",
            len(all_chunks),
            len(documents)
        )

        return all_chunks
