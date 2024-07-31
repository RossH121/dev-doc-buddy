from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from openai import OpenAI

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = "dev-docs"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    return pc.Index(index_name)

def embed_text(text):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Get Pinecone index when needed
def get_pinecone_index():
    return init_pinecone()