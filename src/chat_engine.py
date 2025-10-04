
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict, List
import os


class ChatEngine:
    """Handles chat interactions with documents using LangChain."""
    
    def __init__(self, vector_store, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 500):
        """
        Initialize the chat engine.
        
        Args:
            vector_store: FAISS vector store for document retrieval
            model: OpenAI model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.vector_store = vector_store
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def get_response(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """
        Get a response for the user's query.
        
        Args:
            query: User's question
            chat_history: Previous chat messages
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Get response from chain
            response = self.chain({"question": query})
            
            # Extract source information
            sources = []
            if response.get("source_documents"):
                for doc in response["source_documents"]:
                    source_info = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    sources.append(f"{source_info} (Page {page})\n{content_preview}")
            
            return {
                "answer": response["answer"],
                "sources": sources
            }
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def update_settings(self, temperature: float = None, max_tokens: int = None):
        """
        Update chat engine settings.
        
        Args:
            temperature: New temperature value
            max_tokens: New max tokens value
        """
        if temperature is not None:
            self.temperature = temperature
            self.llm.temperature = temperature
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.llm.max_tokens = max_tokens
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
