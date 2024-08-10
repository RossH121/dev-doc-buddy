# pages/1_Scrape_Docs.py
import streamlit as st
import requests
from urllib.parse import quote
from utils.pinecone_utils import get_pinecone_index, store_chunked_document, clean_content_with_gemini
import json
import nltk
from nltk.tokenize import sent_tokenize
from typing import List
import tiktoken
import logging

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

def scrape_with_scrapingant(url, endpoint, extra_params=None):
    params = {
        "url": url,
        "x-api-key": API_KEY,
    }
    if extra_params:
        params.update(extra_params)
    
    response = requests.get(f"{SCRAPINGANT_API_URL}/{endpoint}", params=params)
    
    if response.status_code == 200:
        credits_used = response.headers.get('Ant-credits-cost', 'Unknown')
        st.sidebar.info(f"API Credits used: {credits_used}")
        data = response.json()
        
        # Clean the content using Gemini 1.5 Pro
        try:
            if isinstance(data, dict) and 'content' in data:
                cleaned_content = clean_content_with_gemini(json.dumps(data['content']))
                if cleaned_content == json.dumps(data['content']):
                    st.warning("Content cleaning did not modify the original content. It may have been blocked due to safety concerns.")
                data['content'] = json.loads(cleaned_content)
            elif isinstance(data, str):
                cleaned_content = clean_content_with_gemini(data)
                if cleaned_content == data:
                    st.warning("Content cleaning did not modify the original content. It may have been blocked due to safety concerns.")
                data = cleaned_content
        except Exception as e:
            st.error(f"Error during content cleaning: {str(e)}")
            logging.error(f"Error during content cleaning: {str(e)}")
        
        # Extract navigation links (if any remain after cleaning)
        navigation_links = []
        if isinstance(data, dict) and 'content' in data:
            for item in data['content']:
                if isinstance(item, dict) and item.get('type') == 'links':
                    navigation_links.extend(item.get('links', []))
                    data['content'].remove(item)
        
        return data, navigation_links
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

def convert_to_markdown(scraped_data):
    markdown_content = ""
    if 'markdown' in scraped_data:
        return scraped_data['markdown']
    
    for item in scraped_data.get('content', []):
        if isinstance(item, dict):
            if item.get('type') == 'heading':
                markdown_content += f"{'#' * item['level']} {item['text']}\n\n"
            elif item.get('type') == 'paragraph':
                markdown_content += f"{item['text']}\n\n"
            elif item.get('type') == 'list':
                for li in item.get('items', []):
                    markdown_content += f"- {li}\n"
                markdown_content += "\n"
            elif item.get('type') == 'code':
                markdown_content += f"```{item.get('language', '')}\n{item['code']}\n```\n\n"
        elif isinstance(item, str):
            markdown_content += f"{item}\n\n"
    
    for snippet in scraped_data.get('code_snippets', []):
        if isinstance(snippet, dict):
            markdown_content += f"```{snippet.get('language', '')}\n{snippet['code']}\n```\n\n"
        elif isinstance(snippet, str):
            markdown_content += f"```\n{snippet}\n```\n\n"
    
    return markdown_content.strip()

st.title("Scrape and Manage Development Docs")

url = st.text_input("Enter the URL of the development docs:")
scrape_type = st.radio("Choose scraping type:", 
                       ["Markdown (LLM-ready)", "AI Extraction"])

if scrape_type == "AI Extraction":
    ai_properties = st.text_input("Enter properties for AI extraction (comma-separated):", 
                                  "title, content, code_snippets")

if st.button("Scrape"):
    if url:
        with st.spinner("Scraping and cleaning documentation... This may take a moment."):
            try:
                if scrape_type == "Markdown (LLM-ready)":
                    result, navigation_links = scrape_with_scrapingant(url, "markdown")
                else:  # AI Extraction
                    result, navigation_links = scrape_with_scrapingant(url, "extract", 
                                                     {"extract_properties": quote(ai_properties)})
                
                if result:
                    st.session_state.scraped_data = result
                    st.session_state.scraped_data['url'] = url
                    markdown_content = convert_to_markdown(result)
                    cleaned_markdown = clean_content_with_gemini(markdown_content)
                    if cleaned_markdown == markdown_content:
                        st.warning("Content cleaning did not modify the markdown content. It may have been blocked due to safety concerns.")
                    st.session_state.scraped_data['markdown'] = cleaned_markdown
                    st.session_state.navigation_links = navigation_links
                    st.success(f"{scrape_type} scraping and cleaning completed!")
                else:
                    st.error("Failed to scrape the content. Please check the URL and try again.")
            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")
                logging.error(f"Scraping error: {str(e)}")
    else:
        st.warning("Please enter a URL before scraping.")

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

    # Save to Pinecone
    if st.button("Save to Pinecone"):
        try:
            pinecone_index = get_pinecone_index()
            
            metadata = {
                'url': st.session_state.scraped_data.get('url', ''),
                'scrape_type': scrape_type,
            }
            
            with st.spinner("Saving to Pinecone... This may take a moment for large documents."):
                store_chunked_document(
                    pinecone_index,
                    st.session_state.scraped_data.get('url', ''),
                    st.session_state.scraped_data['markdown'],
                    metadata
                )
            
            st.success("Data saved to Pinecone!")
        except ValueError as ve:
            st.error(f"Error saving to Pinecone: {str(ve)}")
            st.info("The document might be too large or complex. Try splitting it into smaller sections before saving.")
        except Exception as e:
            st.error(f"Error saving to Pinecone: {str(e)}")
            logging.error(f"Pinecone save error: {str(e)}")
            st.info("An unexpected error occurred. Please check the logs for more details.")

if st.session_state.get('navigation_links'):
    st.subheader("Navigation Links")
    col1, col2 = st.columns(2)
    for i, link in enumerate(st.session_state.navigation_links):
        if isinstance(link, dict):
            (col1 if i % 2 == 0 else col2).markdown(f"- [{link.get('text', '')}]({link.get('url', '')})")
        elif isinstance(link, str):
            (col1 if i % 2 == 0 else col2).markdown(f"- {link}")