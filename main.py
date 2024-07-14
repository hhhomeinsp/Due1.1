import os
import io
import uuid
import json
import re
import docx
import PyPDF2
import pandas as pd
import streamlit as st
import logging
import openai
from functools import partial
from tenacity import retry, wait_random_exponential, stop_after_attempt

from pinecone_integration import initialize_pinecone, PineconeConnection
from file_processing import extract_text_from_file
from utils import get_secret, get_embedding, chat_completion, display_questionnaire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PINECONE_DIMENSION = 1536  # Set this to match your index dimension

def set_page_config():
    st.set_page_config(page_title="DUE: Document Understanding Engine", layout="wide")

def initialize_app():
    openai.api_key = get_secret("OPENAI_API_KEY")
    if not openai.api_key:
        st.error("OpenAI API key is not set. Some features may not work.")
        return None

    pinecone_index = initialize_pinecone()
    if pinecone_index is None:
        st.error("Failed to initialize database connection. Some features will be unavailable.")
        return None

    return PineconeConnection(pinecone_index)

def upload_files_to_kb(pinecone_connection, kb_files):
    if pinecone_connection.test_connection():
        for uploaded_file in kb_files:
            try:
                file_contents = extract_text_from_file(uploaded_file)
                embedding = get_embedding(file_contents)
                doc_id = pinecone_connection.add_document(uploaded_file.name, file_contents, embedding)
                if doc_id:
                    st.sidebar.success(f"Processed {uploaded_file.name}")
                else:
                    st.sidebar.error(f"Failed to process {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
    else:
        st.sidebar.error("Cannot process files: No database connection")

def process_uploaded_form(uploaded_form):
    with st.spinner("Processing questionnaire..."):
        try:
            file_content = extract_text_from_file(uploaded_form)
            questions = process_questionnaire(file_content)
            st.session_state["current_questionnaire"] = {"title": uploaded_form.name, "questions": questions}
            st.sidebar.success("Questionnaire processed successfully! Go to the 'Questionnaires' tab to review and edit.")
        except Exception as e:
            st.sidebar.error(f"An error occurred while processing the questionnaire: {str(e)}")
            logger.exception("Error in questionnaire processing")

def display_kb_documents(pinecone_connection):
    st.header("Knowledge Base Documents")
    if pinecone_connection.test_connection():
        documents = pinecone_connection.get_all_documents()
        if documents:
            for doc in documents:
                with st.expander(f"{doc['title']}"):
                    st.write(doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'])
                    if st.button("Delete", key=f"delete_doc_{doc['id']}"):
                        if pinecone_connection.delete_document(doc['id']): 
                            st.success(f"Document '{doc['title']}' deleted successfully.")
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to delete document '{doc['title']}'.")
        else:
            st.info("No documents available in the Knowledge Base.")
    else:
        st.info("Database connection required to view and manage documents.")

def manage_questionnaire(pinecone_connection):
    st.header("Questionnaire Management")
    if "current_questionnaire" in st.session_state:
        st.subheader(f"Current Questionnaire: {st.session_state['current_questionnaire']['title']}")
        questions = st.session_state["current_questionnaire"].get("questions", [])
        if questions:
            st.subheader("Edit Questionnaire")
            edited_questions = display_questionnaire(questions)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Save Questionnaire"):
                    try:
                        questionnaire_id = pinecone_connection.add_questionnaire(
                            st.session_state["current_questionnaire"]["title"], 
                            edited_questions
                        )
                        if questionnaire_id:
                            st.success(f"Questionnaire saved successfully. ID: {questionnaire_id}")
                            st.session_state["current_questionnaire"]["questions"] = edited_questions
                        else:
                            st.error("Failed to save questionnaire.")
                    except Exception as e:
                        st.error(f"Error saving questionnaire: {str(e)}")
                        logger.exception("Error saving questionnaire")
            with col2:
                if st.button("Generate Report"):
                    generate_report(pinecone_connection, edited_questions)
            with col3:
                if st.button("Create New Questionnaire"):
                    st.session_state["current_questionnaire"] = {
                        "title": "New Questionnaire",
                        "questions": []
                    }
                    st.experimental_rerun()
        else:
            st.warning("The current questionnaire has no questions.")
    else:
        st.info("No current questionnaire to display. Please upload a questionnaire from the sidebar or load a saved one.")
    display_saved_questionnaires(pinecone_connection)

def generate_report(pinecone_connection, questions):
    if pinecone_connection.test_connection():
        report = []
        for question in questions:
            try:
                query_embedding = get_embedding(question['question'])
                similar_docs = pinecone_connection.get_similar_documents(query_embedding)

                context = "\n".join([text for _, _, text, _ in similar_docs])
                prompt = f"""
                Based on the following context, answer the given question. 
                If the context doesn't contain relevant information for the question, state that the information is not available.

                Question: {question['question']}
                Answer: """

                messages = [
                    {"role": "system", "content": "You are a helpful assistant that generates detailed answers based on given questions and context."},
                    {"role": "user", "content": prompt}
                ]

                response = chat_completion(messages)
                answer = response['choices'][0]['message']['content'].strip()

                report.append({
                    "question": question['question'],
                    "answer": answer,
                    "needs_assignment": "information is not available" in answer.lower()
                })
            except Exception as e:
                logger.error(f"Error generating report for question '{question['question']}': {str(e)}")
                st.error(f"Error generating report for question '{question['question']}': {str(e)}")

        st.session_state["current_report"] = report
        display_report(report)
    else:
        st.error("Cannot generate report: No database connection")

def display_report(report):
    st.header("Generated Report")
    for item in report:
        st.subheader(f"Question: {item['question']}")
        st.write(f"Answer: {item['answer']}")
        if item["needs_assignment"]:
            st.warning("This answer needs further assignment.")

    if st.button("Save Report"):
        save_report(report)

def save_report(report):
    report_content = json.dumps(report, indent=4)
    st.download_button(
        label="Download Report",
        data=report_content,
        file_name="generated_report.json",
        mime="application/json"
    )

def display_saved_questionnaires(pinecone_connection):
    st.header("Saved Questionnaires")
    try:
        saved_questionnaires = pinecone_connection.get_all_questionnaires()
        if saved_questionnaires:
            for q in saved_questionnaires:
                with st.expander(f"Questionnaire: {q['title']}"):
                    st.write(f"ID: {q['id']}")
                    st.write("Questions:")
                    for i, question in enumerate(q['questions'], 1):
                        st.write(f"{i}. {question['question']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Load", key=f"load_{q['id']}"):
                            st.session_state["current_questionnaire"] = q
                            st.experimental_rerun()
                    with col2:
                        if st.button("Delete", key=f"delete_{q['id']}"):
                            if pinecone_connection.delete_questionnaire(q['id']): 
                                st.success(f"Questionnaire '{q['title']}' deleted successfully.")
                                st.experimental_rerun()
                            else:
                                st.error(f"Failed to delete questionnaire '{q['title']}'.")
        else:
            st.info("No saved questionnaires found.")
    except Exception as e:
        st.error(f"Error retrieving saved questionnaires: {str(e)}")
        logger.exception("Error retrieving saved questionnaires")

def handle_query(pinecone_connection):
    st.header("Ask a Question")
    query = st.text_input("Enter your question about the documents in the Knowledge Base")
    if st.button("Submit"):
        if pinecone_connection.test_connection():
            try:
                query_embedding = get_embedding(query)
                similar_docs = pinecone_connection.get_similar_documents(query_embedding)

                if similar_docs:
                    st.subheader("Most Relevant Documents")
                    for i, (doc_id, title, text, score) in enumerate(similar_docs, 1):
                        with st.expander(f"{i}. {title} (Similarity: {score:.4f})"):
                            st.write(text[:300] + "..." if len(text) > 300 else text)

                    context = "\n".join([text for _, _, text, _ in similar_docs])
                    prompt = f"""
                    Based on the following context, answer the question. If the answer is not in the context, say "I don't have enough information to answer that question."

                    Context: {context[:3000]}  # Limiting context to 3000 characters

                    Question: {query}
                    Answer: """

                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on given context."},
                        {"role": "user", "content": prompt}
                    ]

                    response = chat_completion(messages)
                    result = response['choices'][0]['message']['content'].strip()

                    st.subheader("Answer")
                    st.write(result)
                else:
                    st.write("No relevant documents found in the Knowledge Base.")
            except Exception as e:
                st.error(f"Error querying the system: {str(e)}")
        else:
            st.error("Cannot perform query: No database connection")

def main():
    set_page_config()
    
    custom_css = """
        <style>
            /* Custom styles for buttons */
            .stButton > button {
                background-color: #2196F3;  /* Changed to a blue color */
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
                background-color: #1976D2;  /* Darker blue for hover state */
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
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)
    st.title("DUE: Document Understanding Engine")

    pinecone_connection = initialize_app()
    if pinecone_connection is None:
        return

    st.sidebar.header("Add Content to Knowledge Base")
    kb_files = st.sidebar.file_uploader("Choose file(s) to upload to Knowledge Base", 
                                        type=["txt", "pdf", "docx", "xlsx", "xls"], 
                                        accept_multiple_files=True)
    if kb_files and st.sidebar.button("Process Knowledge Base File(s)"):
        upload_files_to_kb(pinecone_connection, kb_files)

    st.sidebar.header("Upload Questionnaire/Form")
    uploaded_form = st.sidebar.file_uploader("Choose a form/questionnaire to upload", 
                                             type=["pdf", "docx", "txt", "xlsx", "xls"])
    if uploaded_form and st.sidebar.button("Process Questionnaire"):
        process_uploaded_form(uploaded_form)

    kb_tab, questionnaire_tab, query_tab = st.tabs(["Knowledge Base", "Questionnaires", "Ask a Question"])

    with kb_tab:
        display_kb_documents(pinecone_connection)

    with questionnaire_tab:
        manage_questionnaire(pinecone_connection)

    with query_tab:
        handle_query(pinecone_connection)

if __name__ == "__main__":
    main()
