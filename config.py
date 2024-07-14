import streamlit as st
import openai
import pinecone
from utils import get_secret

def set_page_config():
    st.set_page_config(page_title="DUE: Document Understanding Engine", layout="wide")
    # Apply custom CSS here

def initialize_openai():
    openai.api_key = get_secret("OPENAI_API_KEY")
    return openai.api_key

def initialize_pinecone():
    pinecone_api_key = get_secret("PINECONE_API_KEY")
    pinecone_environment = get_secret("PINECONE_ENVIRONMENT")

    if not pinecone_api_key:
        st.error("Pinecone API key is not set.")
        return None

    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        index_name = get_secret("PINECONE_INDEX_NAME", "thoth")
        
        if index_name not in pinecone.list_indexes():
            # Create index logic here
            pass
        
        return pinecone.Index(index_name)
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None
