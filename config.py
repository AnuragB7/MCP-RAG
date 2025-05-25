import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # OpenAI API Key
    OPENAI_API_KEY = ""

    BASE_URL = ""

    MODEL_NAME = "claude-3-5-sonnet"

    # EMBEDDING_MODEL 
    
    # File processing settings
    MAX_FILE_SIZE_MB = 200  # Maximum file size to process
    CHUNK_SIZE_BYTES = 8192  # For file reading
    MEMORY_THRESHOLD_MB = 30  # Threshold for chunking strategy
    PROCESSING_TIMEOUT_SECONDS = 600  # 10 minutes


    # Milvus settings
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "rag_documents")
    MILVUS_COLLECTION_NAME = "document_embeddings"

    # RAG settings
    # EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
    EMBEDDING_MODEL = "text-embedding-bge-m3"
    USE_OPENAI_EMBEDDINGS = bool(OPENAI_API_KEY)  # Use OpenAI if key available
    VECTOR_DB_PATH = "./data/chroma_db"
    T_SYSTEMS_MAX_BATCH_SIZE = 128

    # Text chunking settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # Search settings
    DEFAULT_SEARCH_RESULTS = 10
    MAX_CONTEXT_LENGTH = 4000
    
    # Streamlit settings
    UPLOAD_FOLDER = "./data/uploads"
    MAX_UPLOAD_SIZE_MB = 200
    PDF_BATCH_SIZE = 5  # Process 5 pages at a time

    # Create directories
    @classmethod
    def setup_directories(cls):
        Path(cls.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Initialize directories
Config.setup_directories()
