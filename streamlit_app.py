# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="Dev Docs Assistant", layout="wide")

st.title("Welcome to Dev Docs Assistant")
st.write("Use the sidebar to navigate between different functionalities.")

# You can add any additional content for the home page here
st.write("This is the main page of your Dev Docs Assistant. From here, you can:")
st.write("- Scrape development documentation")
st.write("- Preview and manage scraped data")
st.write("- Interact with the AI assistant")
st.write("- Manage your Pinecone database")

st.write("Select a page from the sidebar to get started!")