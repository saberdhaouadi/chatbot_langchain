
import streamlit as st
import os
from pathlib import Path
import shutil


def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to the temp directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to saved file
    """
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    # Save file
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


def clear_chat_history():
    """Clear the chat history from session state."""
    st.session_state.messages = []
    if st.session_state.chat_engine:
        st.session_state.chat_engine.clear_memory()


def clear_temp_files():
    """Clear temporary uploaded files."""
    temp_dir = Path("temp_uploads")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)


def format_sources(sources: list) -> str:
    """
    Format source documents for display.
    
    Args:
        sources: List of source documents
        
    Returns:
        Formatted string of sources
    """
    if not sources:
        return "No sources found"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(f"**Source {i}:**\n{source}\n")
    
    return "\n".join(formatted)
