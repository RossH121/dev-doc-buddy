import streamlit as st
from openai import OpenAI
from utils.pinecone_utils import get_pinecone_index, embed_text
import json
import re

st.title("AI Assistant")

pinecone_index = get_pinecone_index()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def cleanup_response(response):
    cleaned = re.sub(r'\n\d+\.\s*$', '', response, flags=re.MULTILINE)
    cleaned = cleaned.rstrip()
    last_line = cleaned.split('\n')[-1]
    if "what would you like to do" in last_line.lower() or "what do you want to do" in last_line.lower():
        cleaned = re.sub(r'\n\d+\.?\s*$', '', cleaned, flags=re.MULTILINE)
    return cleaned

def query_assistant(query, chat_history):
    try:
        query_embedding = embed_text(query)
        search_results = pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        context = []
        for match in search_results.matches:
            context.append(f"Content: {match.metadata['content']}\nSource: {match.metadata['url']}")
        
        context_str = "\n\n".join(context)
        
        system_message = """You are a helpful assistant that provides information based on developer documentation. 
        Use the provided context to answer questions, and include code snippets when relevant. 
        Always cite your sources by referencing the URL provided in the 'Source:' field of each context piece."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Based on the following context, answer this question: {query}\n\nContext:\n{context_str}"}
        ]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": query})
        
        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True
        )
    except Exception as e:
        st.error(f"Error querying assistant: {str(e)}")
        return None

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_chat_input():
    if prompt := st.chat_input("What would you like to know about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = query_assistant(prompt, st.session_state.messages)
            
            if response is None:
                return

            message_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in response:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            cleaned_response = cleanup_response(full_response)
                            message_placeholder.markdown(cleaned_response + "▌")
            except Exception as e:
                st.error(f"Error while processing response: {str(e)}")
                return
            
            final_cleaned_response = cleanup_response(full_response)
            message_placeholder.markdown(final_cleaned_response)
        
        st.session_state.messages.append({"role": "assistant", "content": final_cleaned_response})

# Main chat interface
display_chat_messages()
handle_chat_input()