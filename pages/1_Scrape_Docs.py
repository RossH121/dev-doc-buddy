# pages/1_Scrape_Docs.py
import streamlit as st
import requests
from utils.pinecone_utils import get_pinecone_index, store_chunked_document, clean_content_with_gemini, init_pinecone_assistant, upload_file_to_assistant
import json
import nltk
from nltk.tokenize import sent_tokenize
from typing import List
import tiktoken
import logging
import re
from urllib.parse import urlparse

SCRAPINGANT_API_URL = "https://api.scrapingant.com/v2"
API_KEY = st.secrets["SCRAPINGANT_API_KEY"]

logging.basicConfig(level=logging.INFO)

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Download NLTK data if not already present
download_nltk_data()

# Initialize session state
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None

def scrape_with_scrapingant(url):
    params = {
        "url": url,
        "x-api-key": API_KEY,
    }
    
    response = requests.get(f"{SCRAPINGANT_API_URL}/markdown", params=params)
    
    if response.status_code == 200:
        credits_used = response.headers.get('Ant-credits-cost', 'Unknown')
        st.sidebar.info(f"API Credits used: {credits_used}")
        data = response.json()
        
        # Clean the content using Gemini 1.5 Pro
        try:
            cleaned_content = clean_content_with_gemini(data['markdown'])
            if cleaned_content == data['markdown']:
                st.warning("Content cleaning did not modify the original content. It may have been blocked due to safety concerns.")
            data['markdown'] = cleaned_content
        except Exception as e:
            st.error(f"Error during content cleaning: {str(e)}")
            logging.error(f"Error during content cleaning: {str(e)}")
        
        return data, []  # Returning an empty list for navigation links as they're not available in markdown response
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None, None

def chunk_text(text: str, max_tokens: int = 8000) -> List[str]:
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    tokens = encoding.encode(text)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    
    for token in tokens:
        if current_chunk_tokens + 1 > max_tokens:
            chunks.append(encoding.decode(current_chunk))
            current_chunk = []
            current_chunk_tokens = 0
        current_chunk.append(token)
        current_chunk_tokens += 1
    
    if current_chunk:
        chunks.append(encoding.decode(current_chunk))
    
    return chunks

def truncate_content(content, max_size=39000):
    sentences = sent_tokenize(content)
    truncated_content = ""
    for sentence in sentences:
        if len(json.dumps(truncated_content + sentence)) <= max_size:
            truncated_content += sentence + " "
        else:
            break
    return truncated_content.strip() + "..." if len(truncated_content) < len(content) else truncated_content

st.title("Scrape and Manage Development Docs")

url = st.text_input("Enter the URL of the development docs:")

if st.button("Scrape"):
    if url:
        with st.spinner("Scraping and cleaning documentation... This may take a moment."):
            try:
                result, _ = scrape_with_scrapingant(url)
                
                if result:
                    st.session_state.scraped_data = result
                    st.session_state.scraped_data['url'] = url
                    st.success("Scraping and cleaning completed!")
                else:
                    st.error("Failed to scrape the content. Please check the URL and try again.")
            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")
                logging.error(f"Scraping error: {str(e)}")
    else:
        st.warning("Please enter a URL before scraping.")

def get_clean_filename(url):
    parsed_url = urlparse(url)
    clean_url = re.sub(r'[^\w\-_\.]', '_', parsed_url.netloc + parsed_url.path)
    clean_url = re.sub(r'_+', '_', clean_url)
    return f"{clean_url.strip('_')}.txt"

# Display and Edit Results
if st.session_state.scraped_data is not None:
    st.subheader("Scraped Data")
    
    mode = st.radio("Choose mode:", ["View", "Edit"])
    
    if mode == "View":
        st.markdown(st.session_state.scraped_data['markdown'])
    else:
        edited_content = st.text_area("Edit Content", st.session_state.scraped_data['markdown'], height=500)
        if st.button("Save Edits"):
            st.session_state.scraped_data['markdown'] = edited_content
            st.success("Edits saved!")

    # Choose storage option
    storage_option = st.radio("Choose storage option:", ["Pinecone Index", "Pinecone Assistant File"])

    # Save to Pinecone
    if st.button("Save to Pinecone"):
        try:
            if storage_option == "Pinecone Index":
                pinecone_index = get_pinecone_index()
                
                metadata = {
                    'url': st.session_state.scraped_data.get('url', ''),
                    'scrape_type': 'Markdown (LLM-ready)',
                }
                
                with st.spinner("Saving to Pinecone Index... This may take a moment for large documents."):
                    store_chunked_document(
                        pinecone_index,
                        st.session_state.scraped_data.get('url', ''),
                        st.session_state.scraped_data['markdown'],
                        metadata
                    )
                
                st.success("Data saved to Pinecone Index!")
            else:  # Pinecone Assistant File
                assistant = init_pinecone_assistant("dev-doc-buddy")
                
                content = st.session_state.scraped_data['markdown']
                url = st.session_state.scraped_data.get('url', '')
                
                filename = get_clean_filename(url)
                
                # Prepare the content as a formatted string
                file_content = f"URL: {url}\n\nContent:\n\n{content}"
                
                with st.spinner(f"Uploading file '{filename}' to Pinecone Assistant... This may take a moment for large documents."):
                    response = upload_file_to_assistant(assistant, file_content, filename)
                
                if response and isinstance(response, dict) and 'id' in response:
                    st.success(f"File uploaded to Pinecone Assistant! Filename: {filename}, File ID: {response['id']}")
                else:
                    st.warning("File upload response format was unexpected. Please check the Pinecone console.")
                    st.json(response)  # Display the raw response for debugging

        except Exception as e:
            error_message = f"Error saving to Pinecone: {str(e)}"
            st.error(error_message)
            logging.error(error_message)
            if "too large" in str(e).lower():
                st.info("The document might be too large or complex. Try splitting it into smaller sections before saving.")
            else:
                st.info("An unexpected error occurred. Please check the logs for more details.")