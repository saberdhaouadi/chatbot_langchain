from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict, List
import os


class ChatEngine:
    """Handles chat interactions with documents using LangChain and Anthropic Claude."""
    
    def __init__(self, vector_store, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.0, max_tokens: int = 1024):
        """
        Initialize the chat engine with Anthropic Claude.
        
        Args:
            vector_store: FAISS vector store for document retrieval
            model: Anthropic Claude model to use (claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-haiku-20240307)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.vector_store = vector_store
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        
        # Initialize Claude LLM
        self.llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=api_key
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
        Get a response for the user's query using Claude.
        
        Args:
            query: User's question
            chat_history: Previous chat messages (optional, memory is maintained internally)
            
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
    
    def update_settings(self, temperature: float = None, max_tokens: int = None, model: str = None):
        """
        Update chat engine settings.
        
        Args:
            temperature: New temperature value (0.0 to 1.0)
            max_tokens: New max tokens value
            model: New Claude model to use
        """
        recreate_llm = False
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            recreate_llm = True
        
        if max_tokens is not None and max_tokens != self.max_tokens:
            self.max_tokens = max_tokens
            recreate_llm = True
        
        if model is not None and model != self.model:
            self.model = model
            recreate_llm = True
        
        # Recreate LLM if settings changed
        if recreate_llm:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.llm = ChatAnthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                anthropic_api_key=api_key
            )
            
            # Recreate chain with new LLM
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
