import os
from dotenv import load_dotenv

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit page config
st.set_page_config(
    page_title="Gemma AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #4CAF50;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🤖 LangChain Demo with Gemma Model")
st.write("Ask me anything, and I’ll try my best to help!")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

# Input box
input_text = st.text_input("💬 What’s on your mind?", placeholder="Type your question here...")

# Ollama model setup
llm = OllamaLLM(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


# Chat response
if input_text:
    with st.spinner("🤔 Thinking..."):
        response = chain.invoke({"question": input_text})
    st.success("✅ Answer:")
    st.write(response)

# Sidebar info
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Model in use:** `gemma:2b`")
    st.markdown("Built with [LangChain](https://www.langchain.com/) + [Ollama](https://ollama.ai/)")
