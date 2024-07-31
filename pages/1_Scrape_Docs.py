import streamlit as st
import requests
from urllib.parse import quote
from utils.pinecone_utils import get_pinecone_index, embed_text
import json
from nltk.tokenize import sent_tokenize

SCRAPINGANT_API_URL = "https://api.scrapingant.com/v2"
API_KEY = st.secrets["SCRAPINGANT_API_KEY"]

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
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def truncate_content(content, max_size=39000):  # Leave some room for other metadata
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
    
    if 'links' in scraped_data:
        markdown_content += "### Extracted Links\n"
        for link in scraped_data['links'][:10]:
            if isinstance(link, dict):
                markdown_content += f"[{link.get('text', '')}]({link.get('url', '')})\n"
            elif isinstance(link, str):
                markdown_content += f"{link}\n"
    
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
        with st.spinner("Scraping documentation... This may take a moment."):
            if scrape_type == "Markdown (LLM-ready)":
                result = scrape_with_scrapingant(url, "markdown")
            else:  # AI Extraction
                result = scrape_with_scrapingant(url, "extract", 
                                                 {"extract_properties": quote(ai_properties)})
            
            if result:
                st.session_state.scraped_data = result
                st.session_state.scraped_data['url'] = url
                st.session_state.scraped_data['markdown'] = convert_to_markdown(result)
                st.success(f"{scrape_type} scraping completed!")
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
            
            # Use the markdown content for embedding
            embedding = embed_text(st.session_state.scraped_data['markdown'])
            
            # Prepare metadata
            metadata = {
                'url': st.session_state.scraped_data.get('url', ''),
                'scrape_type': scrape_type,
                'content': truncate_content(st.session_state.scraped_data['markdown'])
            }
            
            # Updated upsert operation
            pinecone_index.upsert(
                vectors=[
                    {
                        "id": st.session_state.scraped_data.get('url', ''),
                        "values": embedding,
                        "metadata": metadata
                    }
                ]
            )
            st.success("Data saved to Pinecone!")
        except Exception as e:
            st.error(f"Error saving to Pinecone: {str(e)}")