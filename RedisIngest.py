import pandas as pd
import numpy as np
import fitz
import os
import redis
from Embedding import MiniLMEmbedder, MPNetEmbedder, InstructorEmbedder
from FAISS.measure import timer, memory

# Redis client
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 384  
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"



class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor with the embedding model"""
        self.embedder = MiniLMEmbedder()
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text chunk"""
        return self.embedder.embed_chunks([text])[0]

    def store_embedding(self, file: str, page: str, chunk: str, embedding: np.ndarray):
        """Store document chunk and its embedding in Redis"""
        key = f"{DOC_PREFIX}{file}:{page}:{hash(chunk)}"
        
        # Convert embedding to bytes for Redis storage
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        # Store the document chunk and its embedding
        redis_client.hset(
            key,
            mapping={
                "text": chunk,
                "file": file,
                "page": page,
                "embedding": embedding_bytes
            }
        )

    @timer
    @memory
    def process_pdfs(self, data_dir: str):
        """Process all PDFs in the given directory"""
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                text_by_page = self.extract_text_from_pdf(pdf_path)
                
                for page_num, text in text_by_page:
                    chunks = self.split_text_into_chunks(text)
                    for chunk in chunks:
                        # Get embedding for chunk
                        embedding = self.get_embedding(chunk)
                        # Store in Redis
                        self.store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=chunk,
                            embedding=embedding
                        )
                print(f" -----> Processed {file_name}")

    @staticmethod
    def extract_text_from_pdf(pdf_path: str):
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page

    @staticmethod
    def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100):
        """Split text into chunks of approximately chunk_size words with overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

def clear_redis_store():
    """Clear the Redis database"""
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def create_hnsw_index():
    """Create the Redis vector similarity index"""
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

def main():
    # Initialize the document processor
    processor = DocumentProcessor()
    
    # Clear existing Redis store
    clear_redis_store()
    
    # Create the vector similarity index
    create_hnsw_index()
    
    # Process PDFs from the data directory
    data_dir = "data"
    (result, memory_used), time_taken = processor.process_pdfs(data_dir)
    
    print("Processing complete!")
    print(f"Total Time: {time_taken:.2f} ms")
    print(f"Total Memory: {memory_used:.2f} KB")

if __name__ == '__main__':
    main()