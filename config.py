# config.py
import streamlit as st
import openai
import pinecone
import logging
from secrets import get_secret

logger = logging.getLogger(__name__)

def set_page_config():
    st.set_page_config(page_title="DUE: Document Understanding Engine", layout="wide")
    # Apply custom CSS here if needed

def initialize_openai():
    openai.api_key = get_secret("OPENAI_API_KEY")
    return openai.api_key

def initialize_pinecone():
    pinecone_api_key = get_secret("PINECONE_API_KEY")
    pinecone_environment = get_secret("PINECONE_ENVIRONMENT")
    pinecone_index_name = get_secret("PINECONE_INDEX_NAME")

    logger.info(f"Pinecone Environment: {pinecone_environment}")
    logger.info(f"Pinecone Index Name: {pinecone_index_name}")

    if not pinecone_api_key:
        logger.error("Pinecone API key is not set.")
        return None

    try:
        logger.info(f"Attempting to initialize Pinecone with environment: {pinecone_environment}")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        
        logger.info(f"Pinecone initialized. Checking for index: {pinecone_index_name}")
        
        # List available indexes
        available_indexes = pinecone.list_indexes()
        logger.info(f"Available Pinecone indexes: {available_indexes}")
        
        if pinecone_index_name not in available_indexes:
            logger.warning(f"Index '{pinecone_index_name}' does not exist. Attempting to create new index.")
            try:
                pinecone.create_index(
                    name=pinecone_index_name,
                    dimension=1536,  # Make sure this matches your embedding dimension
                    metric='cosine'
                )
                logger.info(f"Created new Pinecone index: {pinecone_index_name}")
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {str(e)}")
                return None
        
        logger.info(f"Connecting to Pinecone index: {pinecone_index_name}")
        index = pinecone.Index(pinecone_index_name)
        
        # Test the connection by getting index stats
        stats = index.describe_index_stats()
        logger.info(f"Successfully connected to Pinecone index. Stats: {stats}")
        
        return index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        return None
