import streamlit as st
import logging
from utils import get_embedding, chat_completion

logger = logging.getLogger(__name__)

def display_knowledge_base_tab(pinecone_connection):
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

def display_query_tab(pinecone_connection):
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
                logger.exception("Error in querying the system")
        else:
            st.error("Cannot perform query: No database connection")
