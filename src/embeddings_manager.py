
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Optional
import os


class EmbeddingsManager:
    """Manages embeddings and vector store operations using open-source models."""
    
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embeddings manager with HuggingFace embeddings.
        Uses free, open-source embedding models - no API key required!
        
        Args:
            model: HuggingFace embedding model to use
                   Options:
                   - "sentence-transformers/all-MiniLM-L6-v2" (fast, good quality)
                   - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
                   - "BAAI/bge-small-en-v1.5" (good for retrieval)
        """
        # No API key needed for HuggingFace embeddings!
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ Loaded embedding model: {model}")
    
    def create_vector_store(self, chunks: List, persist_directory: Optional[str] = None) -> FAISS:
        """
        Create a FAISS vector store from document chunks.
        
        Args:
            chunks: List of document chunks
            persist_directory: Optional directory to persist the vector store
            
        Returns:
            FAISS vector store
        """
        if not chunks:
            raise ValueError("No chunks provided to create vector store")
        
        print(f"Creating vector store from {len(chunks)} chunks...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        if persist_directory:
            vector_store.save_local(persist_directory)
            print(f"✅ Vector store saved to {persist_directory}")
        
        return vector_store
    
    def load_vector_store(self, persist_directory: str) -> FAISS:
        """
        Load a persisted FAISS vector store.
        
        Args:
            persist_directory: Directory where vector store is saved
            
        Returns:
            Loaded FAISS vector store
        """
        vector_store = FAISS.load_local(
            persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Vector store loaded from {persist_directory}")
        return vector_store
    
    def add_documents(self, vector_store: FAISS, chunks: List) -> FAISS:
        """
        Add new documents to existing vector store.
        
        Args:
            vector_store: Existing FAISS vector store
            chunks: New document chunks to add
            
        Returns:
            Updated vector store
        """
        vector_store.add_documents(chunks)
        print(f"✅ Added {len(chunks)} new chunks to vector store")
        return vector_store
    
    def similarity_search(
        self, 
        vector_store: FAISS, 
        query: str, 
        k: int = 4
    ) -> List:
        """
        Perform similarity search on vector store.
        
        Args:
            vector_store: FAISS vector store
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        results = vector_store.similarity_search(query, k=k)
        return results
