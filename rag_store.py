import asyncio
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGDocumentStore:
    """RAG-enabled document store with vector similarity search"""
    
    def __init__(self):
        self.config = Config()
        self._initialize_vector_db()
        self._initialize_embeddings()
        self._initialize_text_splitter()
        self.documents_metadata = {}
        logger.info("RAG Document Store initialized")
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.VECTOR_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Vector database initialized at {self.config.VECTOR_DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            if self.config.USE_OPENAI_EMBEDDINGS and self.config.OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(api_key=self.config.OPENAI_API_KEY,base_url=self.config.BASE_URL,model=self.config.EMBEDDING_MODEL)
                self.embed_function = self._openai_embed
                logger.info("Using OpenAI embeddings")
            else:
                self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embed_function = self._sentence_transformer_embed
                logger.info(f"Using SentenceTransformer: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_text_splitter(self):
        """Initialize text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def _sentence_transformer_embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using SentenceTransformer"""
        try:
            embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            raise
    
    async def _openai_embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI with T-Systems batch size limits"""
        try:
            # T-Systems has a batch size limit of 128
            max_batch_size = 100  # Use 100 to be safe
            all_embeddings = []
            
            logger.info(f"Creating embeddings for {len(texts)} texts in batches of {max_batch_size}")
            
            # Process in batches
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                batch_num = i//max_batch_size + 1
                total_batches = (len(texts) + max_batch_size - 1)//max_batch_size
                
                logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                try:
                    # Create embeddings for this batch
                    batch_embeddings = await self.embeddings.aembed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Small delay between batches to avoid rate limiting
                    if i + max_batch_size < len(texts):
                        await asyncio.sleep(0.2)  # 200ms delay
                        
                except Exception as batch_error:
                    logger.error(f"Batch {batch_num} failed: {batch_error}")
                    # Try with even smaller batch size
                    if len(batch) > 50:
                        logger.info("Retrying with smaller batch size...")
                        for j in range(0, len(batch), 50):
                            mini_batch = batch[j:j + 50]
                            mini_embeddings = await self.embeddings.aembed_documents(mini_batch)
                            all_embeddings.extend(mini_embeddings)
                            await asyncio.sleep(0.1)
                    else:
                        raise batch_error
            
            logger.info(f"Successfully created {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    # async def add_document(self, doc_id: str, filename: str, file_type: str, 
    #                       content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    #     """Add document with vector embeddings"""
    #     try:
    #         logger.info(f"Adding document: {filename} ({len(content)} characters)")
            
    #         # Split content into chunks
    #         chunks = self.text_splitter.split_text(content)
    #         logger.info(f"Created {len(chunks)} chunks for {filename}")
            
    #         if not chunks:
    #             return {"error": "No content chunks created"}
            
    #         # Create embeddings
    #         embeddings = await self.embed_function(chunks) if asyncio.iscoroutinefunction(self.embed_function) else self.embed_function(chunks)
            
    #         # Prepare data for ChromaDB
    #         chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
    #         chunk_metadata = []
    #         for i, chunk in enumerate(chunks):
    #             chunk_metadata.append({
    #                 "doc_id": doc_id,
    #                 "filename": filename,
    #                 "file_type": file_type,
    #                 "chunk_index": i,
    #                 "chunk_size": len(chunk),
    #                 "upload_time": datetime.now().isoformat(),
    #                 **metadata
    #             })
            
    #         # Add to vector database
    #         self.collection.add(
    #             embeddings=embeddings,
    #             documents=chunks,
    #             metadatas=chunk_metadata,
    #             ids=chunk_ids
    #         )
            
    #         # Store document metadata
    #         self.documents_metadata[doc_id] = {
    #             "filename": filename,
    #             "file_type": file_type,
    #             "chunk_count": len(chunks),
    #             "total_chars": len(content),
    #             "metadata": metadata,
    #             "upload_time": datetime.now().isoformat()
    #         }
            
    #         logger.info(f"Successfully added {filename} with {len(chunks)} chunks")
    #         return {
    #             "doc_id": doc_id,
    #             "chunks_created": len(chunks),
    #             "status": "success"
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Failed to add document {filename}: {e}")
    #         return {"error": str(e)}


    async def add_document(self, doc_id: str, filename: str, file_type: str, 
                        content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add document with vector embeddings and batched ChromaDB insertion"""
        try:
            logger.info(f"Adding document: {filename} ({len(content)} characters)")
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Created {len(chunks)} chunks for {filename}")
            
            if not chunks:
                return {"error": "No content chunks created"}
            
            # Create embeddings (already batched)
            embeddings = await self.embed_function(chunks) if asyncio.iscoroutinefunction(self.embed_function) else self.embed_function(chunks)
            
            # Prepare data for ChromaDB
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "upload_time": datetime.now().isoformat(),
                    **metadata
                })
            
            # Add to ChromaDB in batches
            await self._add_to_chromadb_batched(embeddings, chunks, chunk_metadata, chunk_ids)
            
            # Store document metadata
            self.documents_metadata[doc_id] = {
                "filename": filename,
                "file_type": file_type,
                "chunk_count": len(chunks),
                "total_chars": len(content),
                "metadata": metadata,
                "upload_time": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully added {filename} with {len(chunks)} chunks")
            return {
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to add document {filename}: {e}")
            return {"error": str(e)}

    async def _add_to_chromadb_batched(self, embeddings: List[List[float]], chunks: List[str], 
                                    chunk_metadata: List[Dict], chunk_ids: List[str]):
        """Add data to ChromaDB in batches to avoid batch size limits"""
        try:
            # ChromaDB batch size limit (use conservative value)
            chromadb_batch_size = 5000
            total_chunks = len(chunks)
            
            logger.info(f"Adding {total_chunks} chunks to ChromaDB in batches of {chromadb_batch_size}")
            
            for i in range(0, total_chunks, chromadb_batch_size):
                end_idx = min(i + chromadb_batch_size, total_chunks)
                
                # Get batch data
                batch_embeddings = embeddings[i:end_idx]
                batch_chunks = chunks[i:end_idx]
                batch_metadata = chunk_metadata[i:end_idx]
                batch_ids = chunk_ids[i:end_idx]
                
                # Add batch to ChromaDB
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_chunks,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                batch_num = i // chromadb_batch_size + 1
                total_batches = (total_chunks + chromadb_batch_size - 1) // chromadb_batch_size
                
                logger.info(f"Added ChromaDB batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                
                # Small delay between batches to be gentle on ChromaDB
                if end_idx < total_chunks:
                    await asyncio.sleep(0.1)
            
            logger.info(f"Successfully added all {total_chunks} chunks to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to add to ChromaDB: {e}")
            raise

    async def semantic_search(self, query: str, n_results: int = 10, 
                             doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform semantic similarity search"""
        try:
            logger.info(f"Performing semantic search: '{query}' (n_results={n_results})")
            
            # Create query embedding
            query_embedding = await self.embed_function([query]) if asyncio.iscoroutinefunction(self.embed_function) else self.embed_function([query])
            
            # Prepare filters
            where_filter = {}
            if doc_ids:
                where_filter["doc_id"] = {"$in": doc_ids}
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, self.collection.count()),
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    similarity_score = 1 - results["distances"][0][i] if results["distances"] else 0
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity_score,
                        "chunk_id": results["ids"][0][i] if "ids" in results else f"chunk_{i}"
                    })
            
            logger.info(f"Found {len(formatted_results)} semantic matches")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of all documents"""
        try:
            total_chunks = self.collection.count()
            return {
                "total_documents": len(self.documents_metadata),
                "total_chunks": total_chunks,
                "documents": self.documents_metadata,
                "vector_db_status": "active",
                "embedding_model": self.config.EMBEDDING_MODEL if not self.config.USE_OPENAI_EMBEDDINGS else "OpenAI"
            }
        except Exception as e:
            logger.error(f"Failed to get document summary: {e}")
            return {"error": str(e)}
    
    def clear_all_documents(self):
        """Clear all documents from the store"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection("documents")
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            self.documents_metadata.clear()
            logger.info("All documents cleared from RAG store")
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            raise