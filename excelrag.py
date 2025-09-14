# app.py - Streamlit RAG Application for Excel File Q&A (Pinecone + OpenRouter)
# Features:
# - User-configurable Pinecone API key and index (select/create)
# - User-configurable OpenRouter API key and free model selection
# - Upload Excel files, chunk/embed/index/retrieve/generate
# - Pure Python implementation for simplicity
# Requirements: pip install streamlit openai pinecone-client sentence-transformers pandas python-dotenv structlog torch
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import logging
import structlog
from typing import List, Dict, Any
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="Excel RAG Q&A Chatbot (Pinecone)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables (fallback for local dev)
load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("excel_rag_pinecone")

# Free OpenRouter models (based on current free tier)
FREE_OPENROUTER_MODELS = [
    "deepseek/deepseek-chat-v3.1:free"  
]

# Session state keys
SESSION_KEYS = {
    "messages": "messages",
    "documents": "documents",  # List of indexed vectors
    "upload_status": "upload_status",
    "pinecone_index": "pinecone_index"
}

def initialize_session_state():
    """Initialize session state."""
    defaults = {
        SESSION_KEYS["messages"]: [],
        SESSION_KEYS["documents"]: [],
        SESSION_KEYS["upload_status"]: None,
        SESSION_KEYS["pinecone_index"]: None
    }
    for key, val in defaults.items():
        st.session_state[key] = st.session_state.get(key, val)

def log_event(event: str, details: Dict[str, Any]):
    logger.info(event, **details)

@st.cache_resource
def get_embedding_model():
    """Load sentence-transformers model (free, local embeddings)."""
    return SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims, fast on CPU

@st.cache_resource
def get_pinecone_client(api_key: str):
    """Initialize Pinecone client."""
    if not api_key:
        st.error("Pinecone API key is required.")
        st.stop()
    pc = Pinecone(api_key=api_key)
    return pc

@st.cache_resource
def get_openrouter_client(api_key: str, model: str):
    """Initialize OpenRouter client."""
    if not api_key:
        st.error("OpenRouter API key is required.")
        st.stop()
    if model not in FREE_OPENROUTER_MODELS:
        st.error(f"Selected model {model} is not in free tier. Choose from available options.")
        st.stop()
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

def create_or_select_index(pc, index_name: str, dimension: int = 384):
    """Create index if not exists, or select existing."""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Free tier compatible
        )
        st.success(f"Created new Pinecone index: {index_name}")
    return pc.Index(index_name)

def process_excel_file(file) -> List[Dict[str, Any]]:
    """Extract text from Excel and prepare for indexing."""
    try:
        df = pd.read_excel(file)
        documents = []
        embedding_model = get_embedding_model()
        for idx, row in df.iterrows():
            # Convert row to text chunk
            text = " ".join(str(val) for val in row.values if pd.notna(val))
            if not text.strip():
                continue
            embedding = embedding_model.encode(text).tolist()
            doc_id = f"doc_{idx}_{file.name}"
            documents.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {"content": text, "source": file.name}
            })
        return documents
    except Exception as e:
        log_event("excel_processing_failed", {"file": file.name, "error": str(e)})
        st.error(f"Failed to process Excel file {file.name}: {str(e)}")
        return []

def index_documents(documents: List[Dict[str, Any]], index):
    """Upsert documents to Pinecone."""
    try:
        index.upsert(vectors=documents)
        log_event("documents_indexed", {"count": len(documents)})
    except Exception as e:
        log_event("indexing_failed", {"error": str(e)})
        st.error(f"Failed to index documents: {str(e)}")

def retrieve_documents(query: str, index, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant documents from Pinecone."""
    try:
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [{"id": match['id'], "content": match['metadata']['content'], "score": match['score']} for match in results['matches']]
    except Exception as e:
        log_event("retrieval_failed", {"query": query, "error": str(e)})
        st.error(f"Failed to retrieve documents: {str(e)}")
        return []

def generate_answer(query: str, documents: List[Dict[str, Any]], llm_client: OpenAI) -> str:
    """Generate answer using RAG pipeline."""
    context = "\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in documents])
    prompt = f"""You are an AI assistant for home care/aged care. Answer the question based on the provided Excel context. If context is insufficient, note limitations.

Question: {query}
Context:
{context}

Answer professionally and concisely:"""
    
    try:
        response = llm_client.chat.completions.create(
            model=llm_client.model,  # Use selected model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log_event("llm_answer_failed", {"query": query, "error": str(e)})
        st.error(f"Failed to generate answer: {str(e)}")
        return "Unable to generate answer due to an error."

def main():
    initialize_session_state()
    
    st.title("📊 Excel RAG Q&A Chatbot (Pinecone + OpenRouter)")
    st.markdown("Upload Excel files and ask questions. Uses free Pinecone for vector store and free OpenRouter models.")
    st.info("**Implementation**: Pure Python for simplicity. Generated on September 14, 2025.")
    
    # Sidebar for user configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Pinecone API Key
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", help="Get from pinecone.io (free tier)")
        
        # Pinecone Index
        index_name = st.text_input("Pinecone Index Name", value="excel-rag-index", help="Existing or new index name")
        if st.button("Connect to Pinecone"):
            if pinecone_api_key:
                try:
                    pc = get_pinecone_client(pinecone_api_key)
                    index = create_or_select_index(pc, index_name)
                    st.session_state[SESSION_KEYS["pinecone_index"]] = index
                    st.success(f"Connected to Pinecone index: {index_name}")
                except Exception as e:
                    st.error(f"Pinecone connection failed: {str(e)}")
            else:
                st.warning("Enter Pinecone API key first.")
        
        # OpenRouter API Key
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password", help="Get from openrouter.ai")
        
        # OpenRouter Model Selection
        selected_model = st.selectbox("Select Free Model", FREE_OPENROUTER_MODELS, help="Free tier models only")
        
        if st.button("Connect to OpenRouter"):
            if openrouter_api_key:
                try:
                    llm_client = get_openrouter_client(openrouter_api_key, selected_model)
                    st.success(f"Connected to OpenRouter with model: {selected_model}")
                except Exception as e:
                    st.error(f"OpenRouter connection failed: {str(e)}")
            else:
                st.warning("Enter OpenRouter API key first.")
        
        st.header("Upload Excel Files")
        uploaded_files = st.file_uploader(
            "Upload Excel files (.xlsx, .xls)",
            type=["xlsx", "xls"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.session_state.get(SESSION_KEYS["pinecone_index"]):
            with st.spinner("Processing and indexing..."):
                index = st.session_state[SESSION_KEYS["pinecone_index"]]
                for file in uploaded_files:
                    documents = process_excel_file(file)
                    if documents:
                        index_documents(documents, index)
                        st.session_state[SESSION_KEYS["documents"]].extend(documents)
                        st.session_state[SESSION_KEYS["upload_status"]] = f"Indexed {file.name}"
                        log_event("file_uploaded", {"file": file.name, "doc_count": len(documents)})
                st.success("Files indexed in Pinecone!")
        
        if st.session_state[SESSION_KEYS["upload_status"]]:
            st.write(st.session_state[SESSION_KEYS["upload_status"]])
        
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                if key in SESSION_KEYS.values():
                    del st.session_state[key]
            initialize_session_state()
            st.rerun()
    
    # Main chat interface
    st.header("Ask Questions")
    for message in st.session_state[SESSION_KEYS["messages"]]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if not st.session_state[SESSION_KEYS["documents"]]:
        st.warning("Configure Pinecone/OpenRouter and upload Excel files to start querying.")
        return
    
    # Get configured clients (assume connected)
    pinecone_index = st.session_state.get(SESSION_KEYS["pinecone_index"])
    if not pinecone_index:
        st.warning("Connect to Pinecone first.")
        return
    
    # For OpenRouter, we'll use a global client; in prod, cache per session
    openrouter_api_key = st.session_state.get("openrouter_api_key", "")  # Store in session if needed
    selected_model = st.session_state.get("selected_model", FREE_OPENROUTER_MODELS[0])
    llm_client = get_openrouter_client(openrouter_api_key or st.sidebar.text_input("OpenRouter API Key (fallback)"), selected_model)
    
    if query := st.chat_input("Ask a question about the Excel data..."):
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.spinner("Retrieving and generating answer..."):
            documents = retrieve_documents(query, pinecone_index)
            answer = generate_answer(query, documents, llm_client)
        
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        st.session_state[SESSION_KEYS["messages"]].append({"role": "user", "content": query})
        st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": answer})
        log_event("question_answered", {"query": query, "doc_count": len(documents)})

if __name__ == "__main__":
    main()
