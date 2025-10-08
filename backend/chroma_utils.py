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
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap= 200)
embedding_function = OpenAIEmbeddings()


#Initialize the embedding store
vectorstore = Chroma(persist_directory='api/data/', embedding_function=embedding_function)

def load_documents(folder_path):
    document = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(filename, folder_path)

        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        
        else:
            print("Unsupported file type")
            continue
        
        document.extend(loader.load())
    
    return document

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_documents(file_path)

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        
        vectorstore.add_documents(splits)
        return True
    
    except Exception as e:
        print(f"Error indexing the document{e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        # Use consistent field name - choose either "field_id" or "file_id"
        docs = vectorstore.get(where={"file_id": file_id})  # Changed to "file_id"
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        if docs['ids']:  # Only delete if documents were found
            vectorstore._collection.delete(where={"file_id": file_id})
            print(f"Successfully deleted {len(docs['ids'])} chunks from Chroma")
            return True
        else:
            print(f"No documents found for file_id {file_id}")
            return True  # Or False, depending on your requirements
    except Exception as e:
        print(f"Error deleting the document with file_id: {file_id} --- {e}")
        return False
        