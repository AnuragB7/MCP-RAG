import asyncio
import chromadb
from chromadb.config import Settings
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import numpy as np
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiVectorStore:
    """Unified vector store supporting both ChromaDB and Milvus"""
    
    def __init__(self):
        self.config = Config()
        self.use_chromadb = self.config.VECTOR_DB_TYPE in ["chromadb", "both"]
        self.use_milvus = self.config.VECTOR_DB_TYPE in ["milvus", "both"]
        
        # Initialize embedding model
        self._initialize_embeddings()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len
        )
        
        # Initialize vector databases
        if self.use_chromadb:
            self._initialize_chromadb()
        
        if self.use_milvus:
            self._initialize_milvus()
        
        # Document metadata store
        self.documents_metadata = {}
        
        logger.info(f"MultiVectorStore initialized with: ChromaDB={self.use_chromadb}, Milvus={self.use_milvus}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            if self.config.USE_OPENAI_EMBEDDINGS and self.config.OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(
                    api_key=self.config.OPENAI_API_KEY,
                    base_url=self.config.BASE_URL.replace('/chat/completions', '').replace('/v1', '') + '/v1'
                )
                self.embed_function = self._openai_embed
                self.embedding_dim = 1536  # OpenAI embedding dimension
                logger.info("Using OpenAI embeddings")
            else:
                self.sentence_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
                self.embed_function = self._sentence_transformer_embed
                self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
                logger.info(f"Using SentenceTransformer: {self.config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMADB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized at {self.config.CHROMADB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_milvus(self):
        """Initialize Milvus"""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.config.MILVUS_HOST,
                port=self.config.MILVUS_PORT,
                user=self.config.MILVUS_USER,
                password=self.config.MILVUS_PASSWORD
            )
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="upload_time", dtype=DataType.VARCHAR, max_length=100)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document chunks with embeddings"
            )
            
            # Create or get collection
            collection_name = self.config.MILVUS_COLLECTION_NAME
            
            if utility.has_collection(collection_name):
                self.milvus_collection = Collection(collection_name)
                logger.info(f"Connected to existing Milvus collection: {collection_name}")
            else:
                self.milvus_collection = Collection(
                    name=collection_name,
                    schema=schema
                )
                
                # Create index for vector search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                
                self.milvus_collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                
                logger.info(f"Created new Milvus collection: {collection_name}")
            
            # Load collection
            self.milvus_collection.load()
            logger.info("Milvus collection loaded and ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            logger.error("Make sure Milvus is running. You can start it with: docker run -p 19530:19530 milvusdb/milvus:latest")
            raise
    
    def _sentence_transformer_embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using SentenceTransformer"""
        try:
            embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            raise
    
    async def _openai_embed(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI with batch size limits"""
        try:
            max_batch_size = self.config.T_SYSTEMS_MAX_BATCH_SIZE
            all_embeddings = []
            
            logger.info(f"Creating embeddings for {len(texts)} texts in batches of {max_batch_size}")
            
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                batch_num = i//max_batch_size + 1
                total_batches = (len(texts) + max_batch_size - 1)//max_batch_size
                
                logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                batch_embeddings = await self.embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                if i + max_batch_size < len(texts):
                    await asyncio.sleep(0.2)
            
            logger.info(f"Successfully created {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    async def add_document(self, doc_id: str, filename: str, file_type: str, 
                          content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add document to both vector databases"""
        try:
            logger.info(f"Adding document: {filename} ({len(content)} characters)")
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Created {len(chunks)} chunks for {filename}")
            
            if not chunks:
                return {"error": "No content chunks created"}
            
            # Create embeddings
            embeddings = await self.embed_function(chunks) if asyncio.iscoroutinefunction(self.embed_function) else self.embed_function(chunks)
            
            # Add to ChromaDB
            if self.use_chromadb:
                await self._add_to_chromadb(doc_id, filename, file_type, chunks, embeddings, metadata)
            
            # Add to Milvus
            if self.use_milvus:
                await self._add_to_milvus(doc_id, filename, file_type, chunks, embeddings, metadata)
            
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
    
    async def _add_to_chromadb(self, doc_id: str, filename: str, file_type: str, 
                              chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any]):
        """Add document to ChromaDB"""
        try:
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
            
            self.chroma_collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadata,
                ids=chunk_ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to add to ChromaDB: {e}")
            raise
    
    async def _add_to_milvus(self, doc_id: str, filename: str, file_type: str, 
                            chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any]):
        """Add document to Milvus"""
        try:
            # Prepare data for Milvus
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            doc_ids = [doc_id] * len(chunks)
            filenames = [filename] * len(chunks)
            file_types = [file_type] * len(chunks)
            chunk_indices = list(range(len(chunks)))
            upload_times = [datetime.now().isoformat()] * len(chunks)
            
            # Insert data
            data = [
                ids,
                doc_ids,
                filenames,
                file_types,
                chunk_indices,
                chunks,
                embeddings,
                upload_times
            ]
            
            self.milvus_collection.insert(data)
            self.milvus_collection.flush()
            
            logger.info(f"Added {len(chunks)} chunks to Milvus")
            
        except Exception as e:
            logger.error(f"Failed to add to Milvus: {e}")
            raise
    
    async def semantic_search(self, query: str, n_results: int = 10, 
                             doc_ids: Optional[List[str]] = None, 
                             use_db: str = "auto") -> List[Dict[str, Any]]:
        """Perform semantic search across vector databases"""
        try:
            logger.info(f"Performing semantic search: '{query}' (n_results={n_results})")
            
            # Create query embedding
            query_embedding = await self.embed_function([query]) if asyncio.iscoroutinefunction(self.embed_function) else self.embed_function([query])
            
            # Determine which database to use
            if use_db == "auto":
                use_db = "milvus" if self.use_milvus else "chromadb"
            
            if use_db == "milvus" and self.use_milvus:
                return await self._search_milvus(query_embedding[0], n_results, doc_ids)
            elif use_db == "chromadb" and self.use_chromadb:
                return await self._search_chromadb(query_embedding, n_results, doc_ids)
            elif use_db == "both":
                # Search both and merge results
                results = []
                if self.use_milvus:
                    milvus_results = await self._search_milvus(query_embedding[0], n_results//2, doc_ids)
                    results.extend(milvus_results)
                if self.use_chromadb:
                    chroma_results = await self._search_chromadb(query_embedding, n_results//2, doc_ids)
                    results.extend(chroma_results)
                
                # Sort by similarity and return top results
                results.sort(key=lambda x: x["similarity_score"], reverse=True)
                return results[:n_results]
            else:
                raise ValueError(f"Invalid database selection: {use_db}")
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _search_milvus(self, query_embedding: List[float], n_results: int, 
                            doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search in Milvus"""
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Apply filters if doc_ids specified
            expr = None
            if doc_ids:
                doc_ids_str = ", ".join([f'"{doc_id}"' for doc_id in doc_ids])
                expr = f"doc_id in [{doc_ids_str}]"
            
            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=n_results,
                expr=expr,
                output_fields=["doc_id", "filename", "file_type", "chunk_index", "content", "upload_time"]
            )
            
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "content": hit.entity.get("content"),
                    "metadata": {
                        "doc_id": hit.entity.get("doc_id"),
                        "filename": hit.entity.get("filename"),
                        "file_type": hit.entity.get("file_type"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "upload_time": hit.entity.get("upload_time")
                    },
                    "similarity_score": 1 - hit.distance,  # Convert distance to similarity
                    "chunk_id": hit.id,
                    "source": "milvus"
                })
            
            logger.info(f"Milvus search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []
    
    async def _search_chromadb(self, query_embedding: List[List[float]], n_results: int, 
                              doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search in ChromaDB"""
        try:
            where_filter = {}
            if doc_ids:
                where_filter["doc_id"] = {"$in": doc_ids}
            
            results = self.chroma_collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, self.chroma_collection.count()),
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    similarity_score = 1 - results["distances"][0][i] if results["distances"] else 0
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity_score,
                        "chunk_id": results["ids"][0][i] if "ids" in results else f"chunk_{i}",
                        "source": "chromadb"
                    })
            
            logger.info(f"ChromaDB search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    async def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of all documents across databases"""
        try:
            summary = {
                "total_documents": len(self.documents_metadata),
                "documents": self.documents_metadata,
                "databases": {}
            }
            
            if self.use_chromadb:
                chroma_count = self.chroma_collection.count()
                summary["databases"]["chromadb"] = {
                    "total_chunks": chroma_count,
                    "status": "active"
                }
            
            if self.use_milvus:
                milvus_count = self.milvus_collection.num_entities
                summary["databases"]["milvus"] = {
                    "total_chunks": milvus_count,
                    "status": "active"
                }
            
            summary["embedding_model"] = self.config.EMBEDDING_MODEL if not self.config.USE_OPENAI_EMBEDDINGS else "OpenAI"
            summary["vector_db_type"] = self.config.VECTOR_DB_TYPE
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get document summary: {e}")
            return {"error": str(e)}
    
    def clear_all_documents(self):
        """Clear all documents from both databases"""
        try:
            if self.use_chromadb:
                self.chroma_client.delete_collection("documents")
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("ChromaDB cleared")
            
            if self.use_milvus:
                self.milvus_collection.drop()
                # Recreate collection
                self._initialize_milvus()
                logger.info("Milvus collection cleared")
            
            self.documents_metadata.clear()
            logger.info("All documents cleared from vector stores")
            
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            raise