
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import os


class DocumentProcessor:
    """Handles loading and processing of PDF and Excel documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, file_paths: List[str]) -> List:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path in file_paths:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    
                elif file_extension in ['.xlsx', '.xls']:
                    loader = UnstructuredExcelLoader(file_path, mode="elements")
                    docs = loader.load()
                    documents.extend(docs)
                    
                else:
                    print(f"Unsupported file type: {file_extension}")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                raise
        
        return documents
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> List:
        """
        Complete pipeline: load and split documents.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_documents(file_paths)
        chunks = self.split_documents(documents)
        return chunks
