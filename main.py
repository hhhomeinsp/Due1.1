import streamlit as st
from config import set_page_config, initialize_openai, initialize_pinecone
from database import PineconeConnection
from document_processing import process_knowledge_base_files
from questionnaire import process_questionnaire, display_questionnaire_tab
from report_generation import display_reports_tab
from ui_components import display_knowledge_base_tab, display_query_tab

def main():
    set_page_config()
    
    openai_api_key = initialize_openai()
    if not openai_api_key:
        st.error("OpenAI API key is not set. Some features may not work.")

    pinecone_index = initialize_pinecone()
    if pinecone_index is None:
        st.error("Failed to initialize database connection. Some features will be unavailable.")
        return

    pinecone_connection = PineconeConnection(pinecone_index)

    # Sidebar
    st.sidebar.header("Add Content to Knowledge Base")
    kb_files = st.sidebar.file_uploader("Choose file(s) to upload to Knowledge Base", 
                                        type=["txt", "pdf", "docx", "xlsx", "xls"], 
                                        accept_multiple_files=True)
    if kb_files and st.sidebar.button("Process Knowledge Base File(s)"):
        process_knowledge_base_files(kb_files, pinecone_connection)

    st.sidebar.header("Upload Questionnaire/Form")
    uploaded_form = st.sidebar.file_uploader("Choose a form/questionnaire to upload", 
                                             type=["pdf", "docx", "txt", "xlsx", "xls"])
    if uploaded_form and st.sidebar.button("Process Questionnaire"):
        process_questionnaire(uploaded_form)

    # Main area tabs
    kb_tab, questionnaire_tab, reports_tab, query_tab = st.tabs(["Knowledge Base", "Questionnaires", "Generated Reports", "Ask a Question"])

    with kb_tab:
        display_knowledge_base_tab(pinecone_connection)

    with questionnaire_tab:
        display_questionnaire_tab(pinecone_connection)

    with reports_tab:
        display_reports_tab(pinecone_connection)

    with query_tab:
        display_query_tab(pinecone_connection)

if __name__ == "__main__":
    main()
