from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initializing the text splitter to create embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_function = OpenAIEmbeddings()

# Initialize the embedding store
vectorstore = Chroma(persist_directory='api/data/', embedding_function=embedding_function)

def load_document(file_path: str) -> List[Document]:
    """Load a single document based on file extension"""
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.html'):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return []
        
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return []

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        # Load the single document
        documents = load_document(file_path)
        
        if not documents:
            print(f"No documents loaded from {file_path}")
            return False

        # Split the documents into chunks
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks")

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        
        # Add to vector store
        vectorstore.add_documents(splits)
        print(f"Successfully indexed document {file_path} with file_id {file_id}")
        return True
    
    except Exception as e:
        print(f"Error indexing the document {file_path}: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        # Use consistent field name
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        if docs['ids']:  # Only delete if documents were found
            vectorstore._collection.delete(where={"file_id": file_id})
            print(f"Successfully deleted {len(docs['ids'])} chunks from Chroma")
            return True
        else:
            print(f"No documents found for file_id {file_id}")
            return True
    except Exception as e:
        print(f"Error deleting the document with file_id: {file_id} --- {e}")
        return False