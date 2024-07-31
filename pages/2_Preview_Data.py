import streamlit as st
from utils.pinecone_utils import get_pinecone_index, embed_text
import json
from nltk.tokenize import sent_tokenize

st.title("Preview and Manage Scraped Data")

pinecone_index = get_pinecone_index()

def truncate_content(content, max_size=39000):  # Leave some room for other metadata
    sentences = sent_tokenize(content)
    truncated_content = ""
    for sentence in sentences:
        if len(json.dumps(truncated_content + sentence)) <= max_size:
            truncated_content += sentence + " "
        else:
            break
    return truncated_content.strip() + "..." if len(truncated_content) < len(content) else truncated_content

if 'scraped_data' in st.session_state and st.session_state.scraped_data is not None:
    scraped_data = st.session_state.scraped_data
    
    st.subheader("Scraped Content")
    st.markdown(scraped_data['markdown'])
    
    st.subheader("Edit Content")
    edited_content = st.text_area("Edit the scraped content:", scraped_data['markdown'], height=300)
    
    if st.button("Save Edits"):
        st.session_state.scraped_data['markdown'] = edited_content
        st.success("Edits saved!")
    
    if st.button("Save to Pinecone"):
        try:
            embedding = embed_text(edited_content)
            
            metadata = {
                'url': scraped_data['url'],
                'scrape_type': scraped_data.get('scrape_type', 'Not specified'),
                'content': truncate_content(edited_content)
            }
            
            pinecone_index.upsert(
                vectors=[
                    {
                        "id": scraped_data['url'],
                        "values": embedding,
                        "metadata": metadata
                    }
                ]
            )
            st.success("Data saved to Pinecone!")
        except Exception as e:
            st.error(f"Error saving to Pinecone: {str(e)}")
    
    st.subheader("View Existing Data in Pinecone")
    if st.button("Check if URL exists in Pinecone"):
        try:
            result = pinecone_index.fetch([scraped_data['url']])
            if scraped_data['url'] in result['vectors']:
                st.warning("This URL already exists in Pinecone. Saving will overwrite the existing data.")
                st.json(result['vectors'][scraped_data['url']])
            else:
                st.info("This URL does not exist in Pinecone yet.")
        except Exception as e:
            st.error(f"Error checking Pinecone: {str(e)}")
    
    st.subheader("Delete from Pinecone")
    if st.button("Delete this URL from Pinecone"):
        try:
            pinecone_index.delete(ids=[scraped_data['url']])
            st.success(f"Data for URL {scraped_data['url']} deleted from Pinecone.")
        except Exception as e:
            st.error(f"Error deleting from Pinecone: {str(e)}")

else:
    st.warning("No scraped data available. Please scrape docs first.")

st.subheader("View Recent Entries in Pinecone")
num_results = st.number_input("Number of recent entries to view", min_value=1, max_value=100, value=5)
if st.button("View Recent Entries"):
    try:
        index_stats = pinecone_index.describe_index_stats()
        index_dimension = index_stats['dimension']
        zero_vector = [0] * index_dimension

        results = pinecone_index.query(vector=zero_vector, top_k=num_results, include_metadata=True)
        for match in results['matches']:
            st.write(f"URL: {match['id']}")
            st.write(f"Score: {match['score']}")
            st.write(f"Scrape Type: {match['metadata'].get('scrape_type', 'Not specified')}")
            st.write("Content Preview:")
            st.write(match['metadata']['content'][:200] + "..." if len(match['metadata']['content']) > 200 else match['metadata']['content'])
            st.write("---")
    except Exception as e:
        st.error(f"Error viewing recent entries: {str(e)}")