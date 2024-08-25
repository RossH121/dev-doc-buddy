import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import numpy as np
import tiktoken
import hashlib
import google.generativeai as genai
import logging
import json
from typing import List, Dict, Any
import tempfile
import os

# Constants
INDEX_NAME = "dev-docs"
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_TOKENS = 8000
MAX_METADATA_SIZE = 40000

@st.cache_resource
def init_pinecone(index_name: str = INDEX_NAME) -> Pinecone.Index:
    """Initialize Pinecone connection."""
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

@st.cache_resource
def init_pinecone_assistant(assistant_name: str):
    """Initialize Pinecone Assistant."""
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.assistant.Assistant(assistant_name)

@st.cache_resource
def init_gemini() -> genai.GenerativeModel:
    """Initialize Google Gemini model."""
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return genai.GenerativeModel(model_name='gemini-1.5-pro')

def get_safety_settings() -> List[Dict[str, str]]:
    """Return safety settings for content generation."""
    return [
        {"category": category, "threshold": "BLOCK_NONE"}
        for category in [
            "HARM_CATEGORY_DANGEROUS",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
    ]

def clean_content_with_gemini(content: str) -> str:
    """Clean and summarize documentation content."""
    model = init_gemini()
    prompt = f"""
    You are an AI assistant specialized in cleaning and summarizing developer documentation. 
    Your task is to remove superfluous content while preserving all important technical details, code snippets, 
    and explanations. Maintain the original structure and formatting of the main content.

    Here's the content to clean:

    {content}

    Please provide the cleaned version of this content.
    """

    try:
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            safety_settings=get_safety_settings()
        )

        return response.candidates[0].content.parts[0].text if response.candidates else content
    except ValueError as e:
        logging.error(f"ValueError in clean_content_with_gemini: {str(e)}")
        return content
    except Exception as e:
        logging.error(f"Unexpected error in clean_content_with_gemini: {str(e)}")
        return content

def embed_text(text: str, max_tokens: int = MAX_TOKENS) -> List[float]:
    """Generate embeddings for the given text."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    chunks = chunk_text(text, max_tokens)
    embeddings = [
        client.embeddings.create(input=chunk, model=EMBEDDING_MODEL).data[0].embedding
        for chunk in chunks
    ]

    weights = [len(encoding.encode(chunk)) for chunk in chunks]
    combined_embedding = np.average(embeddings, axis=0, weights=weights)

    return combined_embedding.tolist()

def truncate_metadata(metadata: Dict[str, Any], max_size: int = MAX_METADATA_SIZE) -> Dict[str, Any]:
    """Truncate metadata to ensure it doesn't exceed max size."""
    while True:
        metadata_str = json.dumps(metadata)
        if len(metadata_str.encode('utf-8')) <= max_size:
            return metadata

        if 'chunk_text' in metadata:
            metadata['chunk_text'] = metadata['chunk_text'][:int(len(metadata['chunk_text']) * 0.9)]
        else:
            for key in list(metadata.keys()):
                if isinstance(metadata[key], str) and len(metadata[key]) > 100:
                    metadata[key] = metadata[key][:int(len(metadata[key]) * 0.9)]
                    break
            else:
                if len(metadata) > 1:
                    metadata.pop(next(iter(metadata)))
                else:
                    raise ValueError("Cannot reduce metadata size below limit.")

def chunk_text(text: str, max_tokens: int) -> List[str]:
    """Chunk text into smaller pieces below the max token count."""
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    tokens = encoding.encode(text)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(encoding.decode(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(encoding.decode(current_chunk))

    return chunks

def generate_document_id(url: str) -> str:
    """Generate a unique document ID based on the URL."""
    return hashlib.md5(url.encode()).hexdigest()

def store_chunked_document(index: Pinecone.Index, url: str, text: str, metadata: Dict[str, Any], max_tokens: int = MAX_TOKENS) -> None:
    """Store each chunk of the document with its embeddings and metadata in the Pinecone index."""
    document_id = generate_document_id(url)
    chunks = chunk_text(text, max_tokens)

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        vector_id = f"{document_id}#chunk{i}"
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_text'] = chunk[:1000]  # Keep the first 1000 characters
        chunk_metadata['content'] = chunk[:1000]  # For backwards compatibility
        chunk_metadata['chunk_index'] = i
        chunk_metadata['total_chunks'] = len(chunks)

        truncated_metadata = truncate_metadata(chunk_metadata)

        try:
            index.upsert(
                vectors=[{"id": vector_id, "values": embedding, "metadata": truncated_metadata}]
            )
        except Exception as e:
            logging.error(f"Error upserting vector {vector_id}: {str(e)}")
            raise

def get_document_chunks(index: Pinecone.Index, url: str) -> Dict[str, Any]:
    """Fetch and return all chunks related to a given document URL."""
    document_id = generate_document_id(url)
    results = index.list(prefix=f'{document_id}#')
    return index.fetch(ids=results)

def get_pinecone_index() -> Pinecone.Index:
    """Get the Pinecone index (for backward compatibility)."""
    return init_pinecone()

def upload_file_to_assistant(assistant, content: str, filename: str):
    """Upload a file to Pinecone Assistant."""
    try:
        # Create a temporary file with the specified filename
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_{filename}') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Upload the file
        response = assistant.upload_file(file_path=temp_file_path)

        # Delete the temporary file
        os.unlink(temp_file_path)

        return response
    except Exception as e:
        raise Exception(f"Error uploading file to Pinecone assistant: {str(e)}")