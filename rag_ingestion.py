# rag/ingestion.py
"""
Document ingestion pipeline.

Handles document processing, chunking, embedding, and indexing.
"""

import io
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import asyncpg
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag.embeddings import BedrockEmbeddings
from db.repositories.documents import DocumentsRepository

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Complete ingestion pipeline for RAG documents.
    
    Steps:
    1. Download from S3
    2. Extract text (supports PDF, TXT, MD, HTML)
    3. Chunk with overlap
    4. Generate embeddings
    5. Store in Postgres with TSVECTOR
    """
    
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        s3_client,
        bucket_name: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "amazon.titan-embed-text-v2:0",
    ):
        self.db_pool = db_pool
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = BedrockEmbeddings(model_id=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.docs_repo = DocumentsRepository(db_pool)
    
    async def ingest_from_s3(
        self,
        s3_key: str,
        source_document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest a document from S3.
        
        Args:
            s3_key: S3 object key
            source_document_id: Logical document identifier
            metadata: Additional metadata to store
            force_reindex: Force reindexing even if document exists
            
        Returns:
            Dict with ingestion stats
        """
        logger.info(f"Ingesting document: {s3_key}")
        
        # 1. Download from S3
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
        file_content = response['Body'].read()
        
        # 2. Extract text
        file_type = self._get_file_type(s3_key)
        text = self._extract_text(file_content, file_type)
        
        if not text or len(text.strip()) < 10:
            raise ValueError(f"No text extracted from {s3_key}")
        
        logger.info(f"Extracted {len(text)} characters from {s3_key}")
        
        # 3. Chunk text
        chunks = self._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # 4. Generate embeddings (batch)
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = await self.embeddings.embed_documents(chunk_texts)
        
        # 5. Determine version
        latest_version = await self.docs_repo.get_latest_version(source_document_id)
        new_version = (latest_version or 0) + 1
        
        # 6. Store in database
        base_metadata = metadata or {}
        base_metadata.update({
            "s3_key": s3_key,
            "file_type": file_type,
            "ingestion_timestamp": datetime.utcnow().isoformat(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        })
        
        chunks_created = 0
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    await self.docs_repo.create_document(
                        conn=conn,
                        source_document_id=source_document_id,
                        chunk_index=idx,
                        text=chunk.page_content,
                        embedding=embedding.tolist(),
                        metadata={
                            **base_metadata,
                            **chunk.metadata,
                            "chunk_index": idx,
                        },
                        version=new_version,
                    )
                    chunks_created += 1
        
        logger.info(
            f"Ingestion complete",
            extra={
                "source_document_id": source_document_id,
                "chunks": chunks_created,
                "version": new_version,
            }
        )
        
        return {
            "source_document_id": source_document_id,
            "chunks_created": chunks_created,
            "version": new_version,
            "text_length": len(text),
        }
    
    def _get_file_type(self, filename: str) -> str:
        """Determine file type from filename."""
        if filename.endswith('.pdf'):
            return 'pdf'
        elif filename.endswith('.txt'):
            return 'txt'
        elif filename.endswith('.md'):
            return 'markdown'
        elif filename.endswith('.html'):
            return 'html'
        else:
            return 'unknown'
    
    def _extract_text(self, content: bytes, file_type: str) -> str:
        """
        Extract text from file content.
        
        Supports:
        - PDF (via PyPDF2)
        - TXT/MD (direct decode)
        - HTML (strip tags, basic)
        """
        if file_type == 'pdf':
            return self._extract_text_from_pdf(content)
        elif file_type in ['txt', 'markdown']:
            return content.decode('utf-8')
        elif file_type == 'html':
            # Basic HTML stripping (use BeautifulSoup for production)
            html_text = content.decode('utf-8')
            # Simple tag removal (replace with proper HTML parser)
            import re
            text = re.sub(r'<[^>]+>', '', html_text)
            return text
        else:
            # Try decoding as text
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"Unable to decode file as text")
                return ""
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise ValueError(f"Failed to extract PDF text: {e}")
    
    def _chunk_text(self, text: str) -> List[Any]:
        """
        Chunk text using LangChain's text splitter.
        
        Uses RecursiveCharacterTextSplitter with configurable
        chunk size and overlap.
        """
        chunks = self.text_splitter.create_documents([text])
        return chunks
    
    async def ingest_text_directly(
        self,
        text: str,
        source_document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest text directly (without S3).
        
        Useful for testing or ingesting from other sources.
        """
        logger.info(f"Ingesting text directly: {source_document_id}")
        
        # Chunk
        chunks = self._chunk_text(text)
        
        # Embed
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = await self.embeddings.embed_documents(chunk_texts)
        
        # Store
        latest_version = await self.docs_repo.get_latest_version(source_document_id)
        new_version = (latest_version or 0) + 1
        
        base_metadata = metadata or {}
        base_metadata.update({
            "ingestion_timestamp": datetime.utcnow().isoformat(),
            "source": "direct_text",
        })
        
        chunks_created = 0
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    await self.docs_repo.create_document(
                        conn=conn,
                        source_document_id=source_document_id,
                        chunk_index=idx,
                        text=chunk.page_content,
                        embedding=embedding.tolist(),
                        metadata={**base_metadata, "chunk_index": idx},
                        version=new_version,
                    )
                    chunks_created += 1
        
        return {
            "source_document_id": source_document_id,
            "chunks_created": chunks_created,
            "version": new_version,
        }


class ChunkingStrategy:
    """
    Alternative chunking strategies.
    
    For advanced use cases:
    - Semantic chunking (split on topic boundaries)
    - Sentence-level chunking
    - Paragraph-level chunking
    - Custom separators
    """
    
    @staticmethod
    def semantic_chunking(text: str, model) -> List[str]:
        """
        Chunk based on semantic similarity.
        
        Uses embeddings to detect topic shifts.
        More sophisticated than fixed-size chunking.
        """
        # TODO: Implement semantic chunking
        # 1. Split into sentences
        # 2. Embed sentences
        # 3. Compute similarity between consecutive sentences
        # 4. Split where similarity drops below threshold
        pass
    
    @staticmethod
    def markdown_aware_chunking(text: str) -> List[str]:
        """
        Chunk markdown respecting structure.
        
        Keeps headers with their content.
        """
        # TODO: Implement markdown-aware chunking
        pass
