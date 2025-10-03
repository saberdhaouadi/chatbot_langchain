import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.chat_engine import ChatEngine
from src.utils import save_uploaded_file, clear_chat_history

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Chatbot",
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
st.title("🤖 Document Chatbot")
st.markdown("Upload PDF and Excel documents to chat with your data using AI")

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
                    
                    # Initialize chat engine
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
    st.subheader("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Max Response Length", 100, 2000, 500, 100)
    
    if st.session_state.chat_engine:
        st.session_state.chat_engine.update_settings(
            temperature=temperature,
            max_tokens=max_tokens
        )

# Main chat interface
if not st.session_state.documents_loaded:
    st.info("👈 Please upload and process documents to start chatting")
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
            with st.spinner("Thinking..."):
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
