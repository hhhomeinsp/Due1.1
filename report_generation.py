import streamlit as st
import asyncio
import logging
from functools import partial
from utils import chat_completion

logger = logging.getLogger(__name__)

async def generate_report(questions, documents, progress_bar):
    context = "\n".join([f"Title: {doc['title']}\n{doc['text']}" for doc in documents])
    
    async def process_question(question):
        try:
            prompt = f"""
            Based on the following context, answer the given question. 
            If the context doesn't contain relevant information for the question, state that the information is not available.

            Context: {context[:3000]}  # Limiting context to 3000 characters

            Question: {question['question']}
            Answer: """
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates detailed answers based on given questions and context."},
                {"role": "user", "content": prompt}
            ]
            
            response = await asyncio.to_thread(partial(chat_completion, messages))
            answer = response['choices'][0]['message']['content'].strip()
            return {
                "question": question['question'],
                "answer": answer,
                "needs_assignment": "information is not available" in answer.lower()
            }
        except Exception as e:
            logger.error(f"Error generating answer for question '{question['question']}': {str(e)}")
            return {
                "question": question['question'],
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "needs_assignment": True
            }

    tasks = [process_question(question) for question in questions]
    report = await asyncio.gather(*tasks)

    for i, _ in enumerate(report):
        progress_bar.progress((i + 1) / len(questions))

    logger.info(f"Report generation complete. Total questions processed: {len(report)}")
    return report

def display_reports_tab(pinecone_connection):
    st.header("Generated Reports")
    reports = pinecone_connection.get_all_reports()
    if reports:
        for report in reports:
            with st.expander(f"Report: {report['title']}"):
                for i, qa in enumerate(report['report'], 1):
                    with st.expander(f"Q{i}: {qa['question']}"):
                        st.write("Answer:", qa['answer'])
                        if qa['needs_assignment']:
                            if st.button(f"Assign for Manual Answer", key=f"assign_{report['id']}_{i}"):
                                st.info("This feature will be implemented in the future.")
                
                if st.button("Delete Report", key=f"delete_report_{report['id']}"):
                    if pinecone_connection.delete_report(report['id']):
                        st.success(f"Report '{report['title']}' deleted successfully.")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete report '{report['title']}'.")
    else:
        st.info("No reports available. Generate a report from a processed questionnaire to see it here.")
