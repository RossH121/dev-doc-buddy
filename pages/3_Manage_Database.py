import streamlit as st
from utils.pinecone_utils import get_pinecone_index, embed_text
import json
from nltk.tokenize import sent_tokenize

st.title("Manage Pinecone Database")

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

if st.button("View Database Stats"):
    stats = pinecone_index.describe_index_stats()
    st.write(stats)

st.subheader("Add Test Data")
test_url = st.text_input("Enter a URL for the test data:")
test_content = st.text_area("Enter test content to add to the database:")
scrape_type = st.radio("Choose scrape type for test data:", ["Markdown (LLM-ready)", "AI Extraction"])

if st.button("Add Test Data"):
    if test_url and test_content:
        try:
            embedding = embed_text(test_content)
            truncated_content = truncate_content(test_content)
            metadata = {
                'url': test_url,
                'scrape_type': scrape_type,
                'content': truncated_content
            }
            pinecone_index.upsert(
                vectors=[
                    {
                        "id": test_url,
                        "values": embedding,
                        "metadata": metadata
                    }
                ]
            )
            st.success(f"Test data added with URL: {test_url}")
        except Exception as e:
            st.error(f"Error adding test data: {str(e)}")
    else:
        st.warning("Please enter both URL and content before adding test data.")

st.subheader("View Data")
num_results = st.number_input("Number of results to view", min_value=1, max_value=100, value=10)
if st.button("View Data"):
    try:
        # Get the dimension of the index
        index_stats = pinecone_index.describe_index_stats()
        index_dimension = index_stats['dimension']

        # Create a zero vector with the correct dimension
        zero_vector = [0] * index_dimension

        results = pinecone_index.query(vector=zero_vector, top_k=num_results, include_metadata=True)
        for match in results['matches']:
            st.write(f"URL: {match['id']}")
            st.write(f"Score: {match['score']}")
            
            st.subheader("Content")
            st.write(match['metadata']['content'][:500] + "..." if len(match['metadata']['content']) > 500 else match['metadata']['content'])
            
            st.write(f"Scrape Type: {match['metadata'].get('scrape_type', 'Not specified')}")
            
            st.write("---")
    except Exception as e:
        st.error(f"Error viewing data: {str(e)}")

st.subheader("Delete Data")
delete_url = st.text_input("Enter the URL of the data to delete:")
if st.button("Delete Data"):
    if delete_url:
        try:
            pinecone_index.delete(ids=[delete_url])
            st.success(f"Data with URL {delete_url} deleted successfully")
        except Exception as e:
            st.error(f"Error deleting data: {str(e)}")
    else:
        st.warning("Please enter a URL to delete.")

st.subheader("Clear All Data")
if st.button("Clear All Data"):
    confirm = st.checkbox("I understand this will delete all data in the index.")
    if confirm:
        try:
            # Get all vector IDs
            index_stats = pinecone_index.describe_index_stats()
            total_vector_count = index_stats['total_vector_count']
            
            # Delete all vectors in batches
            batch_size = 1000
            for i in range(0, total_vector_count, batch_size):
                results = pinecone_index.query(vector=[0]*index_stats['dimension'], top_k=batch_size, include_metadata=False)
                ids_to_delete = [match['id'] for match in results['matches']]
                pinecone_index.delete(ids=ids_to_delete)
            
            st.success("All data cleared from the index.")
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")
    else:
        st.warning("Please confirm that you want to clear all data.")