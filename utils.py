import streamlit as st
import openai
import docx
import PyPDF2
import pandas as pd
import io
import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

def get_secret(key, default=None):
    return st.secrets.get(key, default)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        logger.error(f"Error in get_embedding: {str(e)}")
        raise

def chat_completion(messages, model="gpt-4o"):
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return response
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise ValueError(f"Failed to get response from AI model: {str(e)}")

def extract_text_from_file(file):
    try:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'txt':
            return file.getvalue().decode("utf-8")
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(file.getvalue()))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
            return "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(file.getvalue()))
            return df.to_string(index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        raise ValueError(f"Error extracting text from file: {str(e)}")
