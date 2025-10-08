import streamlit as st
import requests
import json
import os
from typing import List, Dict

# FastAPI backend URL
BASE_URL = "http://127.0.0.1:8000"

def init_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []

def call_chat_api(question: str, model: str = "gpt-4o-mini") -> Dict:
    """Call the chat endpoint"""
    url = f"{BASE_URL}/chat"
    payload = {
        "question": question,
        "model": model,
        "session_id": st.session_state.session_id  # Fixed: session_id instead of sessionid
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling chat API: {e}")
        # Print more details about the error
        try:
            error_detail = response.json()
            st.error(f"Error details: {error_detail}")
        except:
            pass
        return None

def upload_document(file) -> bool:
    """Upload document to backend"""
    url = f"{BASE_URL}/upload_doc"
    
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(url, files=files)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {e}")
        return False

def get_documents() -> List[Dict]:
    """Get list of all documents"""
    url = f"{BASE_URL}/list_documents"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching documents: {e}")
        return []

def delete_document(file_id: int) -> bool:
    """Delete document by file_id"""
    url = f"{BASE_URL}/delete-doc"
    payload = {"file_id": file_id}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting document: {e}")
        return False

def main():
    st.set_page_config(
        page_title="RAG Chat Application",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("🤖 Nuera Chat Application")
    st.markdown("Chat with your documents using AI-powered retrieval")
    
    # Sidebar for document management
    with st.sidebar:
        # Model selection
        st.subheader("Model Settings")
        model_option = st.selectbox(
            "Select AI Model",
            ["gpt-4o", "gpt-4o-mini"],  # Fixed: only your available models
            index=1  # Default to gpt-4o-mini
        )
        st.session_state.selected_model = model_option

        st.header("📁 Document Management")
        
        # Document upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'html'],
            help="Supported formats: PDF, DOCX, HTML"
        )
        
        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")
            
            if st.button("Upload & Index Document", type="primary"):
                with st.spinner("Uploading and indexing document..."):
                    if upload_document(uploaded_file):
                        st.success("✅ Document uploaded and indexed successfully!")
                        # Refresh documents list
                        st.session_state.documents = get_documents()
                    else:
                        st.error("❌ Failed to upload document")
        
        st.divider()
        
        # Document list section
        st.subheader("Stored Documents")
        
        if st.button("Refresh Documents"):
            st.session_state.documents = get_documents()
        
        if not st.session_state.documents:
            st.info("No documents uploaded yet")
        else:
            for doc in st.session_state.documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc['filename']}")
                    st.caption(f"ID: {doc['id']} | Uploaded: {doc['uploaded_timestamp']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc['id']}", help="Delete document"):
                        if delete_document(doc['id']):
                            st.success(f"Document {doc['id']} deleted!")
                            st.session_state.documents = get_documents()
                            st.rerun()
        
        st.divider()
        
        # Session info
        st.subheader("Session Info")
        if st.session_state.session_id:
            st.write(f"**Session ID:**")
            st.code(st.session_state.session_id)
            if st.button("New Session"):
                st.session_state.session_id = None
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No active session - start chatting to create one")
        
        st.divider()

    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "human":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "human", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = call_chat_api(
                        question=user_input, 
                        model=st.session_state.selected_model
                    )
                    
                    if response:
                        answer = response["answer"]
                        st.session_state.session_id = response["session_id"]
                        
                        # Add AI response to chat history
                        st.session_state.chat_history.append({"role": "ai", "content": answer})
                        
                        st.write(answer)
                    else:
                        st.error("Failed to get response from AI")
    
    with col2:
        st.header("ℹ️ Info")
        
        st.subheader("How to use:")
        st.markdown("""
        1. 📁 Upload documents in the sidebar
        2. 💬 Ask questions about your documents
        3. 🔄 Use different AI models
        4. 🗑️ Manage documents as needed
        """)
        
        st.divider()
        
        st.subheader("Current Stats:")
        st.metric("Documents", len(st.session_state.documents))
        st.metric("Chat Messages", len(st.session_state.chat_history))
        
        if st.session_state.session_id:
            st.metric("Session", "Active")
        else:
            st.metric("Session", "Inactive")
        
        st.divider()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()