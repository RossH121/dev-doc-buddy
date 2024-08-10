#pages/3_AI_Assistant.py
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from utils.pinecone_utils import get_pinecone_index, embed_text
import re
import tiktoken

# Constants for GPT-4o mini
GPT4O_MINI_MODEL = "gpt-4o-mini"
GPT4O_MINI_CONTEXT_WINDOW = 128000
GPT4O_MINI_OUTPUT_LIMIT = 16384

# Constants for Claude 3.5 Sonnet
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
CLAUDE_CONTEXT_WINDOW = 200000
CLAUDE_OUTPUT_LIMIT = 8192

def num_tokens_from_messages(messages, model, system_message=""):
    """
    Calculate the number of tokens from a list of messages, including the system message.
    """
    if model == GPT4O_MINI_MODEL:
        encoding = tiktoken.encoding_for_model("gpt-4")  # Use GPT-4 encoding as a close approximation
        num_tokens = len(encoding.encode(system_message))  # Include system message tokens
        for message in messages:
            num_tokens += 4  # Each message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens -= 1  # Omit role token if name is present
        num_tokens += 2  # Every reply is primed with <im_start>assistant
    elif model == CLAUDE_MODEL:
        # Claude uses different tokenization, this is an approximation
        num_tokens = len(system_message.split()) * 1.3  # Include system message tokens
        num_tokens += sum(len(message['content'].split()) for message in messages) * 1.3
    else:
        raise ValueError(f"Unsupported model: {model}")
    return int(num_tokens)

# Initialize Streamlit app
st.title("AI Assistant")
pinecone_index = get_pinecone_index()
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Model selection
model_option = st.selectbox(
    "Select AI Model",
    (GPT4O_MINI_MODEL, CLAUDE_MODEL),
    index=0
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "output_tokens" not in st.session_state:
    st.session_state.output_tokens = 0

# Set up sidebar structure
st.sidebar.markdown("### Token Usage")

# Initialize session state for sidebar placeholders
if "sidebar_placeholders" not in st.session_state:
    st.session_state.sidebar_placeholders = {
        "total_progress": st.sidebar.empty(),
        "total_text": st.sidebar.empty(),
        "total_percentage": st.sidebar.empty(),
        "output_progress": st.sidebar.empty(),
        "output_text": st.sidebar.empty(),
        "output_percentage": st.sidebar.empty()
    }

def cleanup_response(response):
    """
    Clean up the assistant's response.
    """
    cleaned = re.sub(r'\n\d+\.\s*$', '', response, flags=re.MULTILINE).rstrip()
    last_line = cleaned.split('\n')[-1]
    if "what would you like to do" in last_line.lower() or "what do you want to do" in last_line.lower():
        cleaned = re.sub(r'\n\d*\.?\s*$', '', cleaned, flags=re.MULTILINE)
    return cleaned

def query_assistant(query, chat_history, model):
    """
    Query the AI assistant with the user's question.
    """
    try:
        # Generate query embedding and fetch search results from Pinecone
        query_embedding = embed_text(query)
        search_results = pinecone_index.query(vector=query_embedding, top_k=20, include_metadata=True)

        # Collate context from search results
        context = []
        for match in search_results.matches:
            chunk_text = match.metadata.get('chunk_text', match.metadata.get('content', 'No content available'))
            url = match.metadata.get('url', 'No URL available')
            chunk_index = match.metadata.get('chunk_index', 'N/A')
            total_chunks = match.metadata.get('total_chunks', 'N/A')
            context.append(f"Content: {chunk_text}\nSource: {url}\nChunk: {chunk_index + 1}/{total_chunks}" if chunk_index != 'N/A' else f"Content: {chunk_text}\nSource: {url}")

        context_str = "\n\n".join(context)

        # Define system message
        system_message = """You are a helpful assistant that provides guidance and generates code based on developer documentation.
        Use the provided context to answer questions, and include full code snippets when relevant or requested.
        Always cite your sources by referencing the URL provided in the 'Source:' field of each context piece.
        When referencing information from a specific chunk, mention the chunk number and total chunks if available.
        If asked to generate code, ensure it's well-commented and follows SOLID coding principles.
        You always start your responses with a section called "Thinking" where you provide an in-depth explanation of your thought process and reasoning for a specific task. 
        You always use lateral thinking skills when problem-solving to find an innovative and efficient way to solve a particular problem."""

        # Combine context and query into a single user message
        combined_query = f"Based on the following context, answer this question: {query}\n\nContext:\n{context_str}"

        # Prepare messages for API call, ensuring alternation between user and assistant
        messages = []
        for i, msg in enumerate(chat_history):
            if msg["role"] in ["user", "assistant"]:
                if i == 0 or msg["role"] != messages[-1]["role"]:
                    messages.append(msg)

        # Add the combined query as the last user message
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": combined_query})
        else:
            # If the last message is already a user message, update its content
            messages[-1]["content"] += f"\n\nAdditional context:\n{combined_query}"

        # Calculate total tokens including system message
        total_tokens = num_tokens_from_messages(messages, model, system_message)
        st.session_state.total_tokens = total_tokens

        # Calculate available tokens for response
        safety_margin = 100

        if model == GPT4O_MINI_MODEL:
            available_tokens = GPT4O_MINI_CONTEXT_WINDOW - total_tokens - safety_margin
            max_tokens = min(available_tokens, GPT4O_MINI_OUTPUT_LIMIT)

            # Truncate context if necessary to fit the model's requirements
            while total_tokens > GPT4O_MINI_CONTEXT_WINDOW - max_tokens - safety_margin:
                if len(messages) > 1:
                    messages.pop(0)  # Remove the oldest message
                else:
                    messages[0]["content"] = messages[0]["content"][:int(len(messages[0]["content"]) * 0.9)]  # Truncate context
                total_tokens = num_tokens_from_messages(messages, model, system_message)

            # Call the OpenAI model for a response
            return openai_client.chat.completions.create(
                model=GPT4O_MINI_MODEL,
                temperature=0.9,
                messages=[{"role": "system", "content": system_message}] + messages,
                stream=True,
                max_tokens=max_tokens
            )
        elif model == CLAUDE_MODEL:
            available_tokens = CLAUDE_CONTEXT_WINDOW - total_tokens - safety_margin
            max_tokens = min(available_tokens, CLAUDE_OUTPUT_LIMIT)

            # Truncate context if necessary to fit the model's requirements
            while total_tokens > CLAUDE_CONTEXT_WINDOW - max_tokens - safety_margin:
                if len(messages) > 1:
                    messages.pop(0)  # Remove the oldest message
                else:
                    messages[0]["content"] = messages[0]["content"][:int(len(messages[0]["content"]) * 0.9)]  # Truncate context
                total_tokens = num_tokens_from_messages(messages, model, system_message)

            # Call the Anthropic model for a response
            return anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                messages=messages,
                system=system_message,
                temperature=0.9,
                max_tokens=max_tokens,
                extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                stream=True
            )
    except Exception as e:
        if hasattr(e, 'status_code') and e.status_code == 400:
            st.error(f"API Error: {str(e)}")
        else:
            st.error(f"Error querying assistant: {str(e)}")
        return None

def display_chat_messages():
    """
    Display chat messages from session state.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def update_sidebar(model):
    """
    Update sidebar with token usage information.
    """
    if model == GPT4O_MINI_MODEL:
        context_window = GPT4O_MINI_CONTEXT_WINDOW
        output_limit = GPT4O_MINI_OUTPUT_LIMIT
    elif model == CLAUDE_MODEL:
        context_window = CLAUDE_CONTEXT_WINDOW
        output_limit = CLAUDE_OUTPUT_LIMIT

    # Helper function to safely update placeholders
    def safe_update(key, value_func):
        if key in st.session_state.sidebar_placeholders:
            value_func(st.session_state.sidebar_placeholders[key])

    # Total Tokens
    safe_update("total_progress", lambda p: p.progress(min(1.0, st.session_state.total_tokens / context_window)))
    safe_update("total_text", lambda p: p.markdown(f"**Total Tokens:** {st.session_state.total_tokens:,} / {context_window:,}"))
    safe_update("total_percentage", lambda p: p.markdown(f"**Total Usage:** {min(100, (st.session_state.total_tokens / context_window) * 100):.2f}%"))

    # Output Tokens
    safe_update("output_progress", lambda p: p.progress(min(1.0, st.session_state.output_tokens / output_limit)))
    safe_update("output_text", lambda p: p.markdown(f"**Output Tokens:** {st.session_state.output_tokens:,} / {output_limit:,}"))
    safe_update("output_percentage", lambda p: p.markdown(f"**Output Usage:** {min(100, (st.session_state.output_tokens / output_limit) * 100):.2f}%"))

def handle_chat_input(model):
    """
    Handle user input and process the assistant's response.
    """
    if prompt := st.chat_input("What would you like to know about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = query_assistant(prompt, st.session_state.messages, model)

            if response is None:
                return

            message_placeholder = st.empty()
            full_response = ""
            accumulated_output_tokens = 0

            try:
                if model == GPT4O_MINI_MODEL:
                    for chunk in response:
                        if chunk.choices:
                            content = chunk.choices[0].delta.content
                            if content:
                                full_response += content
                                chunk_tokens = num_tokens_from_messages([{"role": "assistant", "content": content}], model)
                                accumulated_output_tokens += chunk_tokens
                                cleaned_response = cleanup_response(full_response)
                                message_placeholder.markdown(cleaned_response + "▌")
                elif model == CLAUDE_MODEL:
                    for chunk in response:
                        if chunk.type == "content_block_delta":
                            content = chunk.delta.text
                            if content:
                                full_response += content
                                chunk_tokens = num_tokens_from_messages([{"role": "assistant", "content": content}], model)
                                accumulated_output_tokens += chunk_tokens
                                cleaned_response = cleanup_response(full_response)
                                message_placeholder.markdown(cleaned_response + "▌")
                        elif chunk.type == "message_stop":
                            break
            except Exception as e:
                st.error(f"Error while processing response: {str(e)}")
                return

            final_cleaned_response = cleanup_response(full_response)
            message_placeholder.markdown(final_cleaned_response)

        st.session_state.messages.append({"role": "assistant", "content": final_cleaned_response})

        # Update session state token counts
        st.session_state.output_tokens += accumulated_output_tokens
        
        # Recalculate total tokens
        system_message = """You are a helpful assistant that provides guidance and generates code based on developer documentation..."""  # Full system message here
        st.session_state.total_tokens = num_tokens_from_messages(st.session_state.messages, model, system_message)

        # Update sidebar with final token counts
        update_sidebar(model)

def clear_chat():
    """
    Clear chat messages and reset session state.
    """
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.session_state.output_tokens = 0
    update_sidebar(model_option)  # Update sidebar after clearing chat

if st.button("Clear Chat", key="clear_chat_button"):
    clear_chat()

# Main chat interface
update_sidebar(model_option)  # Initial update of sidebar
display_chat_messages()
handle_chat_input(model_option)

