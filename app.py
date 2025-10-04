               
import sys
import os
from pathlib import Path

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.chat_engine import ChatEngine
from src.utils import save_uploaded_file, clear_chat_history

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Chatbot with Claude",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Title and description
st.title("🤖 Document Chatbot with Claude")
st.markdown("Upload PDF and Excel documents to chat with your data using Anthropic's Claude AI")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload PDF or Excel files"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
        for file in uploaded_files:
            st.write(f"• {file.name}")
    
    # Process documents button
    if st.button("🔄 Process Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one document")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = save_uploaded_file(uploaded_file)
                        file_paths.append(file_path)
                    
                    # Process documents
                    processor = DocumentProcessor()
                    documents = processor.load_documents(file_paths)
                    chunks = processor.split_documents(documents)
                    
                    # Create embeddings and vector store
                    embeddings_manager = EmbeddingsManager()
                    vector_store = embeddings_manager.create_vector_store(chunks)
                    
                    # Initialize chat engine with Claude
                    st.session_state.chat_engine = ChatEngine(vector_store)
                    st.session_state.documents_loaded = True
                    
                    st.success(f"✅ Successfully processed {len(documents)} documents!")
                    st.info(f"Created {len(chunks)} text chunks for search")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()
    
    # Settings
    st.divider()
    st.subheader("⚙️ Claude Settings")
    
    # Model selection
    claude_model = st.selectbox(
        "Claude Model",
        options=[
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        index=0,
        help="Choose the Claude model to use"
    )
    
    temperature = st.slider(
        "Temperature", 
        0.0, 1.0, 0.0, 0.1,
        help="Higher values make output more random, lower values more focused"
    )
    
    max_tokens = st.slider(
        "Max Response Length", 
        100, 4096, 1024, 100,
        help="Maximum number of tokens in the response"
    )
    
    if st.session_state.chat_engine:
        st.session_state.chat_engine.update_settings(
            temperature=temperature,
            max_tokens=max_tokens,
            model=claude_model
        )
    
    # Display model info
    st.divider()
    with st.expander("ℹ️ About Claude Models"):
        st.markdown("""
        **Claude 3.5 Sonnet**: Best balance of intelligence and speed (recommended)
        
        **Claude 3 Opus**: Most capable model for complex tasks
        
        **Claude 3 Sonnet**: Fast and capable for most tasks
        
        **Claude 3 Haiku**: Fastest model for simple tasks
        """)

# Main chat interface
if not st.session_state.documents_loaded:
    st.info("👈 Please upload and process documents to start chatting")
    
    # Display API key status
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not anthropic_key:
        st.warning("⚠️ Missing API key!")
        st.markdown("""
        Please ensure your `.env` file contains:
        - `ANTHROPIC_API_KEY` - Get it from [Anthropic Console](https://console.anthropic.com/)
        
        Note: Embeddings use free open-source models (no API key needed!)
        """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Claude is thinking..."):
                try:
                    response = st.session_state.chat_engine.get_response(
                        prompt,
                        st.session_state.messages
                    )
                    
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response.get("sources"):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}:** {source}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
