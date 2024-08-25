import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
import re
import tiktoken
import logging
from typing import List, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GPT4O_MINI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
PINECONE_ASSISTANT_MODEL = "pinecone-assistant"
PINECONE_ASSISTANT_NAME = "dev-doc-buddy"
INDEX_NAME = "dev-docs"
EMBEDDING_MODEL = "text-embedding-3-large"

MODEL_CONFIGS = {
    GPT4O_MINI_MODEL: {"context_window": 128000, "output_limit": 16384},
    CLAUDE_MODEL: {"context_window": 200000, "output_limit": 8192},
    PINECONE_ASSISTANT_MODEL: {"context_window": float('inf'), "output_limit": float('inf')}
}

@st.cache_resource
def initialize_pinecone():
    api_key = st.secrets.get("PINECONE_API_KEY")
    if not api_key:
        st.error("Pinecone API key not found. Please set it in your Streamlit secrets.")
        return None
    try:
        pc = Pinecone(api_key=api_key)
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=3072,
                metric="cosine",
                spec=Pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
            )
        return pc
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        st.error(f"Failed to initialize Pinecone. Please check your API key and try again.")
        return None

def embed_text(text: str, client: OpenAI) -> List[float]:
    """Generate embeddings for the given text."""
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

class AIAssistant:
    def __init__(self, model: str):
        self.model = model
        self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        self.pinecone_instance = initialize_pinecone()
        self.pinecone_index = self.pinecone_instance.Index(INDEX_NAME)
        self.pinecone_assistant = self.get_assistant()

    def get_assistant(self):
        if self.model == PINECONE_ASSISTANT_MODEL:
            try:
                return self.pinecone_instance.assistant.describe_assistant(PINECONE_ASSISTANT_NAME)
            except Exception as e:
                logger.error(f"Error connecting to Pinecone Assistant: {e}")
                st.error(f"Error connecting to Pinecone Assistant. Please check your configuration and try again.")
                return None
        return None

    def num_tokens_from_messages(self, messages: List[Dict[str, str]], system_message: str = "") -> int:
        """Calculate the number of tokens from a list of messages, including the system message."""
        if self.model == PINECONE_ASSISTANT_MODEL:
            return sum(len(message['content'].split()) for message in messages) * 1.3

        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(system_message))

        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens -= 1

        num_tokens += 2
        return int(num_tokens)

    def cleanup_response(self, response: str) -> str:
        """Clean up the assistant's response."""
        cleaned = re.sub(r'\n\d+\.\s*$', '', response, flags=re.MULTILINE).rstrip()
        last_line = cleaned.split('\n')[-1]
        if "what would you like to do" in last_line.lower() or "what do you want to do" in last_line.lower():
            cleaned = re.sub(r'\n\d*\.?\s*$', '', cleaned, flags=re.MULTILINE)
        return cleaned

    def query_assistant(self, query: str, chat_history: List[Dict[str, str]]) -> Any:
        try:
            if self.model == PINECONE_ASSISTANT_MODEL:
                chat_context = [Message(content=m["content"], role=m["role"]) for m in chat_history]
                chat_context.append(Message(content=query, role="user"))
                return self.pinecone_assistant.chat_completions(messages=chat_context, stream=True)
            # For other models, use Pinecone index for context retrieval
            query_embedding = embed_text(query, self.openai_client)
            search_results = self.pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True)
            context = self._collate_context(search_results)
            system_message = """You are a helpful assistant that provides guidance and generates code based on developer documentation.
            Use the provided context to answer questions, and include full code snippets when relevant or requested.
            Always cite your sources by referencing the URL provided in the 'Source:' field of each context piece.
            When referencing information from a specific chunk, mention the chunk number and total chunks if available.
            If asked to generate code, ensure it's well-commented and follows SOLID coding principles.
            You always start your responses with a section called "Thinking" where you provide an in-depth explanation of your thought process and reasoning for a specific task. 
            You always use lateral thinking skills when problem-solving to find an innovative and efficient way to solve a particular problem."""
            messages = self._prepare_messages(chat_history, query, context)
            total_tokens = self.num_tokens_from_messages(messages, system_message)
            st.session_state.total_tokens = total_tokens
            config = MODEL_CONFIGS[self.model]
            available_tokens = config["context_window"] - total_tokens - 100
            max_tokens = min(available_tokens, config["output_limit"])
            messages = self._truncate_messages(messages, system_message, max_tokens, config["context_window"])
            if self.model == GPT4O_MINI_MODEL:
                return self.openai_client.chat.completions.create(
                    model=GPT4O_MINI_MODEL,
                    temperature=0.9,
                    messages=[{"role": "system", "content": system_message}] + messages,
                    stream=True,
                    max_tokens=max_tokens
                )
            elif self.model == CLAUDE_MODEL:
                claude_messages = messages
                if context:
                    claude_messages.insert(0, {"role": "user", "content": f"Here's some context for the conversation:\n\n{context}"})
                    claude_messages.insert(1, {"role": "assistant", "content": "Thank you for providing the context. I'll keep this information in mind while answering questions."})
                return self.anthropic_client.messages.create(
                    model=CLAUDE_MODEL,
                    messages=claude_messages,
                    system=system_message,
                    max_tokens=max_tokens,
                    temperature=0.9,
                    stream=True
                )
        except Exception as e:
            logger.error(f"Error querying assistant: {str(e)}")
            st.error(f"Error querying assistant: {str(e)}")
            return None

    def _collate_context(self, search_results: Any) -> str:
        context = []
        for match in search_results.matches:
            chunk_text = match.metadata.get('chunk_text', match.metadata.get('content', 'No content available'))
            url = match.metadata.get('url', 'No URL available')
            chunk_index = match.metadata.get('chunk_index', 'N/A')
            total_chunks = match.metadata.get('total_chunks', 'N/A')
            context.append(f"Content: {chunk_text}\nSource: {url}\nChunk: {chunk_index + 1}/{total_chunks}" if chunk_index != 'N/A' else f"Content: {chunk_text}\nSource: {url}")
        return "\n\n".join(context)

    def _prepare_messages(self, chat_history: List[Dict[str, str]], query: str, context: str = "") -> List[Dict[str, str]]:
        messages = []
        for i, msg in enumerate(chat_history):
            if msg["role"] in ["user", "assistant"] and (i == 0 or msg["role"] != chat_history[i-1]["role"]):
                messages.append(msg)
        
        if context and self.model != CLAUDE_MODEL:
            combined_query = f"Based on the following context, answer this question: {query}\n\nContext:\n{context}"
        else:
            combined_query = query
        
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": combined_query})
        else:
            messages[-1]["content"] += f"\n\nAdditional question: {combined_query}"
        
        return messages

    def _truncate_messages(self, messages: List[Dict[str, str]], system_message: str, max_tokens: int, context_window: int) -> List[Dict[str, str]]:
        while self.num_tokens_from_messages(messages, system_message) > context_window - max_tokens - 100:
            if len(messages) > 1:
                messages.pop(0)
            else:
                messages[0]["content"] = messages[0]["content"][:int(len(messages[0]["content"]) * 0.9)]
        return messages

def handle_chat_input(assistant: AIAssistant):
    """Handle user input and process the assistant's response."""
    if prompt := st.chat_input("What would you like to know about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = assistant.query_assistant(prompt, st.session_state.messages)
            if response is None:
                return
            message_placeholder = st.empty()
            full_response = ""
            accumulated_output_tokens = 0

            try:
                for chunk in response:
                    if assistant.model == PINECONE_ASSISTANT_MODEL:
                        if chunk.choices:
                            content = chunk.choices[0].delta.content
                            if content:
                                full_response += content
                                cleaned_response = assistant.cleanup_response(full_response)
                                message_placeholder.markdown(cleaned_response + "▌")
                    else:
                        content = ""
                        if assistant.model == GPT4O_MINI_MODEL:
                            content = chunk.choices[0].delta.content
                        elif assistant.model == CLAUDE_MODEL:
                            content = chunk.delta.text if chunk.type == "content_block_delta" else ''

                        if content:
                            full_response += content
                            cleaned_response = assistant.cleanup_response(full_response)
                            message_placeholder.markdown(cleaned_response + "▌")

                            chunk_tokens = assistant.num_tokens_from_messages([{"role": "assistant", "content": content}], "")
                            accumulated_output_tokens += chunk_tokens

                        if assistant.model == CLAUDE_MODEL and chunk.type == "message_stop":
                            break

            except Exception as e:
                logger.error(f"Error while processing response: {str(e)}")
                st.error("Error while processing response. Please try again.")
                return

            final_cleaned_response = assistant.cleanup_response(full_response)
            message_placeholder.markdown(final_cleaned_response)
        st.session_state.messages.append({"role": "assistant", "content": final_cleaned_response})
        
        if assistant.model != PINECONE_ASSISTANT_MODEL:
            st.session_state.output_tokens += accumulated_output_tokens
            st.session_state.total_tokens = assistant.num_tokens_from_messages(st.session_state.messages, "")
        
        update_sidebar(assistant.model)
        st.rerun()

def update_sidebar(model: str):
    """Update sidebar with token usage information."""
    config = MODEL_CONFIGS[model]

    def safe_update(key: str, value_func: callable):
        if key in st.session_state.sidebar_placeholders:
            value_func(st.session_state.sidebar_placeholders[key])

    safe_update("total_progress", lambda p: p.progress(min(1.0, st.session_state.total_tokens / config["context_window"])))
    safe_update("total_text", lambda p: p.markdown(f"**Total Tokens:** {st.session_state.total_tokens:,} / {config['context_window']:,}"))
    safe_update("total_percentage", lambda p: p.markdown(f"**Total Usage:** {min(100, (st.session_state.total_tokens / config['context_window']) * 100):.2f}%"))

    safe_update("output_progress", lambda p: p.progress(min(1.0, st.session_state.output_tokens / config["output_limit"])))
    safe_update("output_text", lambda p: p.markdown(f"**Output Tokens:** {st.session_state.output_tokens:,} / {config['output_limit']:,}"))
    safe_update("output_percentage", lambda p: p.markdown(f"**Output Usage:** {min(100, (st.session_state.output_tokens / config['output_limit']) * 100):.2f}%"))

def clear_chat():
    """Clear chat messages and reset session state."""
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.session_state.output_tokens = 0
    update_sidebar(st.session_state.model_option)
    st.rerun()

def display_chat_messages():
    """Display chat messages from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.title("AI Assistant")

    model_option = st.selectbox(
        "Select AI Model",
        (GPT4O_MINI_MODEL, CLAUDE_MODEL, PINECONE_ASSISTANT_MODEL),
        index=0
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "output_tokens" not in st.session_state:
        st.session_state.output_tokens = 0
    if "model_option" not in st.session_state:
        st.session_state.model_option = model_option

    st.sidebar.markdown("### Token Usage")

    if "sidebar_placeholders" not in st.session_state:
        st.session_state.sidebar_placeholders = {
            "total_progress": st.sidebar.empty(),
            "total_text": st.sidebar.empty(),
            "total_percentage": st.sidebar.empty(),
            "output_progress": st.sidebar.empty(),
            "output_text": st.sidebar.empty(),
            "output_percentage": st.sidebar.empty()
        }

    assistant = AIAssistant(model_option)

    update_sidebar(model_option)
    display_chat_messages()
    handle_chat_input(assistant)

    if st.button("Clear Chat", key="clear_chat_button"):
        clear_chat()

if __name__ == "__main__":
    main()