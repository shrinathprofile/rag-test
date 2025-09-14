
# app.py - Streamlit RAG Application for PDF File Q&A (Pinecone + OpenRouter)
# Features:
# - Upload PDF files, extract text, chunk, embed, index, and query
# - User-configurable Pinecone API key and index (select/create)
# - User-configurable OpenRouter API key and free model selection
# - User-selectable chunking strategy (Fixed-Length/Sentence-Based) with parameters
# - User-customizable system prompt with role examples
# - Pure Python implementation for simplicity
# - Author: Grok 4 (xAI) - Updated on September 14, 2025, 11:25 AM ACST
# Requirements: pip install streamlit openai pinecone-client sentence-transformers PyPDF2 python-dotenv structlog torch
# Run: streamlit run app.py

import streamlit as st
import PyPDF2
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
import re

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="PDF RAG Q&A Chatbot (Pinecone)",
    page_icon="📄",
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
logger = structlog.get_logger("pdf_rag_pinecone")

# Free OpenRouter models (based on current free tier, September 2025)
FREE_OPENROUTER_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",  # Strong reasoning
    "meta-llama/llama-3.3-8b-instruct:free"  # Llama 3.1 8B
]

# Chunking strategies
CHUNKING_STRATEGIES = ["Fixed-Length", "Sentence-Based"]

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are an AI assistant for home care/aged care. Answer questions based on the provided PDF context in a professional and concise manner. If context is insufficient, note limitations."

# Example system prompts
EXAMPLE_PROMPTS = [
    "Act as a clinician analyzing patient care plans to provide precise medical insights.",
    "Act as a new employee searching policies and procedures to understand organizational guidelines.",
    "Act as a compliance officer ensuring regulatory adherence to standards like HIPAA or NDIS."
]

# Session state keys
SESSION_KEYS = {
    "messages": "messages",
    "documents": "documents",
    "upload_status": "upload_status",
    "pinecone_index": "pinecone_index",
    "openrouter_api_key": "openrouter_api_key",
    "selected_model": "selected_model",
    "system_prompt": "system_prompt",
    "chunking_strategy": "chunking_strategy",
    "chunk_length": "chunk_length",
    "max_sentences": "max_sentences"
}

def initialize_session_state():
    """Initialize session state."""
    defaults = {
        SESSION_KEYS["messages"]: [],
        SESSION_KEYS["documents"]: [],
        SESSION_KEYS["upload_status"]: None,
        SESSION_KEYS["pinecone_index"]: None,
        SESSION_KEYS["openrouter_api_key"]: "",
        SESSION_KEYS["selected_model"]: FREE_OPENROUTER_MODELS[0],
        SESSION_KEYS["system_prompt"]: DEFAULT_SYSTEM_PROMPT,
        SESSION_KEYS["chunking_strategy"]: CHUNKING_STRATEGIES[0],
        SESSION_KEYS["chunk_length"]: 500,
        SESSION_KEYS["max_sentences"]: 5
    }
    for key, val in defaults.items():
        st.session_state[key] = st.session_state.get(key, val)

def log_event(event: str, details: Dict[str, Any]):
    logger.info(event, **details)

@st.cache_resource
def get_embedding_model():
    """Load sentence-transformers model (free, local embeddings)."""
    return SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims, CPU-friendly

@st.cache_resource
def get_pinecone_client(api_key: str):
    """Initialize Pinecone client."""
    if not api_key:
        st.error("Pinecone API key is required.")
        st.stop()
    try:
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        st.error(f"Pinecone connection failed: {str(e)}")
        st.stop()

@st.cache_resource
def get_openrouter_client(api_key: str):
    """Initialize OpenRouter client."""
    if not api_key:
        st.error("OpenRouter API key is required.")
        st.stop()
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return client
    except Exception as e:
        st.error(f"OpenRouter connection failed: {str(e)}")
        st.stop()

def create_or_select_index(pc, index_name: str, dimension: int = 384):
    """Create index if not exists, or select existing."""
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Free tier
            )
            st.success(f"Created new Pinecone index: {index_name}")
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Failed to create/select index: {str(e)}")
        st.stop()

def chunk_text_fixed_length(text: str, max_length: int) -> List[str]:
    """Chunk text by fixed character length."""
    chunks = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def chunk_text_sentence_based(text: str, max_sentences: int) -> List[str]:
    """Chunk text by sentence count."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    sentence_count = 0
    for sentence in sentences:
        current_chunk.append(sentence)
        sentence_count += 1
        if sentence_count >= max_sentences:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = []
            sentence_count = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return [chunk for chunk in chunks if chunk]

def process_pdf_file(file, strategy: str, chunk_length: int, max_sentences: int) -> List[Dict[str, Any]]:
    """Extract text from PDF and prepare for indexing."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + " "
        
        # Apply chunking strategy
        if strategy == "Fixed-Length":
            chunks = chunk_text_fixed_length(text, chunk_length)
        else:  # Sentence-Based
            chunks = chunk_text_sentence_based(text, max_sentences)
        
        documents = []
        embedding_model = get_embedding_model()
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            embedding = embedding_model.encode(chunk).tolist()
            doc_id = f"doc_{idx}_{file.name}"
            documents.append({
                "id": doc_id,
                "values": embedding,
                "metadata": {"content": chunk, "source": file.name}
            })
        return documents
    except Exception as e:
        log_event("pdf_processing_failed", {"file": file.name, "error": str(e)})
        st.error(f"Failed to process PDF file {file.name}: {str(e)}")
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
        return [{"id": match['id'], "content": match['metadata']['content'], "score": match['score'], "source": match['metadata']['source']} for match in results['matches']]
    except Exception as e:
        log_event("retrieval_failed", {"query": query, "error": str(e)})
        st.error(f"Failed to retrieve documents: {str(e)}")
        return []

def generate_answer(query: str, documents: List[Dict[str, Any]], llm_client: OpenAI, model: str, system_prompt: str) -> str:
    """Generate answer using RAG pipeline."""
    context = "\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in documents])
    prompt = f"""System: {system_prompt}

Question: {query}
Context:
{context}

Answer professionally and concisely:"""
    
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
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
    
    st.title("📄 PDF RAG Q&A Chatbot (Pinecone + OpenRouter)")
    st.markdown("Upload PDF files and ask questions. Customize chunking strategy and system prompt. Uses free Pinecone for vector store and free OpenRouter models.")
    st.info("**Implementation**: Pure Python, free tier Pinecone/OpenRouter. Generated on September 14, 2025, 11:25 AM ACST.")
    
    # Sidebar for user configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Pinecone API Key
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", help="Get from pinecone.io (free tier)")
        
        # Pinecone Index
        index_name = st.text_input("Pinecone Index Name", value="pdf-rag-index", help="Existing or new index name")
        if st.button("Connect to Pinecone"):
            if pinecone_api_key:
                pc = get_pinecone_client(pinecone_api_key)
                index = create_or_select_index(pc, index_name)
                st.session_state[SESSION_KEYS["pinecone_index"]] = index
                st.success(f"Connected to Pinecone index: {index_name}")
            else:
                st.warning("Enter Pinecone API key first.")
        
        # OpenRouter API Key
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password", help="Get from openrouter.ai")
        if openrouter_api_key != st.session_state[SESSION_KEYS["openrouter_api_key"]]:
            st.session_state[SESSION_KEYS["openrouter_api_key"]] = openrouter_api_key
        
        # OpenRouter Model Selection
        selected_model = st.selectbox("Select Free Model", FREE_OPENROUTER_MODELS, index=FREE_OPENROUTER_MODELS.index(st.session_state[SESSION_KEYS["selected_model"]]), help="Free tier models only")
        if selected_model != st.session_state[SESSION_KEYS["selected_model"]]:
            st.session_state[SESSION_KEYS["selected_model"]] = selected_model
        
        # Chunking Strategy and Parameters
        st.header("Chunking Strategy")
        chunking_strategy = st.selectbox("Chunking Strategy", CHUNKING_STRATEGIES, index=CHUNKING_STRATEGIES.index(st.session_state[SESSION_KEYS["chunking_strategy"]]))
        if chunking_strategy != st.session_state[SESSION_KEYS["chunking_strategy"]]:
            st.session_state[SESSION_KEYS["chunking_strategy"]] = chunking_strategy
        
        if chunking_strategy == "Fixed-Length":
            chunk_length = st.slider("Max Chunk Length (characters)", min_value=100, max_value=1000, value=st.session_state[SESSION_KEYS["chunk_length"]], step=50)
            st.session_state[SESSION_KEYS["chunk_length"]] = chunk_length
        else:  # Sentence-Based
            max_sentences = st.slider("Max Sentences per Chunk", min_value=1, max_value=10, value=st.session_state[SESSION_KEYS["max_sentences"]], step=1)
            st.session_state[SESSION_KEYS["max_sentences"]] = max_sentences
        
        # System Prompt
        st.header("System Prompt")
        st.markdown("Customize the AI's role. Examples:")
        for example in EXAMPLE_PROMPTS:
            st.markdown(f"- {example}")
        system_prompt = st.text_area("System Prompt", value=st.session_state[SESSION_KEYS["system_prompt"]], height=100)
        if system_prompt != st.session_state[SESSION_KEYS["system_prompt"]]:
            st.session_state[SESSION_KEYS["system_prompt"]] = system_prompt
        
        # File Upload
        st.header("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Upload PDF files (.pdf)",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.session_state.get(SESSION_KEYS["pinecone_index"]):
            with st.spinner("Processing and indexing..."):
                index = st.session_state[SESSION_KEYS["pinecone_index"]]
                for file in uploaded_files:
                    documents = process_pdf_file(file, st.session_state[SESSION_KEYS["chunking_strategy"]], 
                                               st.session_state[SESSION_KEYS["chunk_length"]], 
                                               st.session_state[SESSION_KEYS["max_sentences"]])
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
        st.warning("Configure Pinecone/OpenRouter, set prompt, and upload PDF files to start querying.")
        return
    
    # Validate configurations
    pinecone_index = st.session_state.get(SESSION_KEYS["pinecone_index"])
    if not pinecone_index:
        st.warning("Connect to Pinecone first.")
        return
    
    openrouter_api_key = st.session_state.get(SESSION_KEYS["openrouter_api_key"])
    if not openrouter_api_key:
        st.warning("Enter OpenRouter API key first.")
        return
    
    selected_model = st.session_state.get(SESSION_KEYS["selected_model"])
    system_prompt = st.session_state.get(SESSION_KEYS["system_prompt"])
    llm_client = get_openrouter_client(openrouter_api_key)
    
    if query := st.chat_input("Ask a question about the PDF data..."):
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.spinner("Retrieving and generating answer..."):
            documents = retrieve_documents(query, pinecone_index)
            answer = generate_answer(query, documents, llm_client, selected_model, system_prompt)
        
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        st.session_state[SESSION_KEYS["messages"]].append({"role": "user", "content": query})
        st.session_state[SESSION_KEYS["messages"]].append({"role": "assistant", "content": answer})
        log_event("question_answered", {"query": query, "doc_count": len(documents)})

if __name__ == "__main__":
    main()
