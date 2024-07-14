import streamlit as st
import logging
from utils import extract_text_from_file, get_embedding

logger = logging.getLogger(__name__)

def process_knowledge_base_files(kb_files, pinecone_connection):
    if pinecone_connection.test_connection():
        for uploaded_file in kb_files:
            try:
                file_contents = extract_text_from_file(uploaded_file)
                embedding = get_embedding(file_contents)
                doc_id = pinecone_connection.add_document(uploaded_file.name, file_contents, embedding)
                if doc_id:
                    st.sidebar.success(f"Processed {uploaded_file.name}")
                    logger.info(f"Successfully processed and added document: {uploaded_file.name}")
                else:
                    st.sidebar.error(f"Failed to process {uploaded_file.name}")
                    logger.error(f"Failed to add document: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
                logger.exception(f"Error processing {uploaded_file.name}")
    else:
        st.sidebar.error("Cannot process files: No database connection")
        logger.error("Failed to process files due to no database connection")

def extract_document_content(file_content):
    try:
        return file_content
    except Exception as e:
        logger.error(f"Error extracting document content: {str(e)}")
        raise ValueError("Failed to extract document content. Please check the file format.")
