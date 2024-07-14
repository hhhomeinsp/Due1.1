import streamlit as st
import openai
import pinecone
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PINECONE_DIMENSION = 1536  # Set this to match your index dimension

def get_secret(key, default=None):
    return st.secrets.get(key, default)

def set_page_config():
    st.set_page_config(page_title="DUE: Document Understanding Engine", layout="wide")
    
    # Apply custom CSS
    st.markdown("""
        <style>
            /* Custom styles for buttons */
            .stButton > button {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
                transition: background-color 0.3s;
            }
        
            .stButton > button:hover {
                background-color: #1976D2;
            }

            /* Custom styles for dropdowns (select boxes) */
            .stSelectbox > div > div {
                background-color: #2F3336;
                color: white;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
            }

            .stSelectbox > div > div:hover {
                border-color: #4CAF50;
            }

            /* Style for the dropdown options */
            .stSelectbox > div > div > ul {
                background-color: #2F3336;
                color: white;
            }

            .stSelectbox > div > div > ul > li:hover {
                background-color: #3A3F42;
            }

            /* Custom styles for expanders */
            .streamlit-expanderHeader {
                background-color: #2F3336;
                color: white;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
                padding: 10px;
                margin-bottom: 10px;
            }

            .streamlit-expanderHeader:hover {
                background-color: #3A3F42;
            }

            /* Custom styles for text inputs */
            .stTextInput > div > div > input {
                background-color: #2F3336;
                color: white;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
                padding: 10px;
            }

            .stTextInput > div > div > input:focus {
                border-color: #4CAF50;
                box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
            }

            /* Ensure text color is white for all inputs */
            .stTextInput, .stSelectbox, .stTextArea {
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

def initialize_openai():
    openai.api_key = get_secret("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key is not set.")
    return openai.api_key

def initialize_pinecone():
    pinecone_api_key = get_secret("PINECONE_API_KEY")
    pinecone_environment = get_secret("PINECONE_ENVIRONMENT")

    if not pinecone_api_key:
        logger.error("Pinecone API key is not set.")
        return None

    try:
        logger.info(f"Attempting to initialize Pinecone with environment: {pinecone_environment}")
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        
        index_name = get_secret("PINECONE_INDEX_NAME", "thoth")
        logger.info(f"Checking for index: {index_name}")
        
        if index_name not in pinecone.list_indexes():
            logger.warning(f"Index '{index_name}' does not exist. Attempting to create new index.")
            try:
                pinecone.create_index(
                    name=index_name,
                    dimension=PINECONE_DIMENSION,
                    metric='cosine'
                )
                logger.info(f"Created new Pinecone index: {index_name}")
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {str(e)}")
                return None
        
        logger.info(f"Successfully connected to Pinecone index: {index_name}")
        return pinecone.Index(index_name)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        return None
