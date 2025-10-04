
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Optional
import os


class EmbeddingsManager:
    """Manages embeddings and vector store operations."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the embeddings manager.
        
        Args:
            model: OpenAI embedding model to use
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key
        )
    
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
        
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        if persist_directory:
            vector_store.save_local(persist_directory)
        
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
