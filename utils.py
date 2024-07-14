import openai
import logging
import json
import re
import asyncio
from functools import partial
from tenacity import retry, wait_random_exponential, stop_after_attempt
import streamlit as st

logger = logging.getLogger(__name__)

# Utility function to get secrets
def get_secret(key, default=None):
    return st.secrets.get(key, default)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response['data'][0]['embedding']
    return embedding  # The embedding should naturally be 1536 dimensions

def chat_completion(messages, model="gpt-4"):
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return response
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise ValueError(f"Failed to get response from AI model: {str(e)}")

def display_questionnaire(questions, prefix=""):
    edited_questions = []
    for i, question in enumerate(questions):
        try:
            if not isinstance(question, dict) or 'question' not in question:
                st.warning(f"Invalid question format at {prefix}{i+1}. Skipping.")
                continue

            with st.expander(f"{prefix}{i+1}. {question['question'][:50]}..."):
                edited_question = {}
                edited_question['question'] = st.text_input("Question", question.get("question", ""), key=f"q_{prefix}{i}")
                question_type = question.get("type", "text").lower()
                edited_question['type'] = st.selectbox("Type", 
                                                       ["text", "number", "date", "yes/no", "multiple choice", "file upload"],
                                                       index=["text", "number", "date", "yes/no", "multiple choice", "file upload"].index(question_type),
                                                       key=f"type_{prefix}{i}")
                edited_question['instructions'] = st.text_area("Instructions", question.get("instructions", ""), key=f"instr_{prefix}{i}")
                
                if edited_question['type'] == "multiple choice":
                    options = st.text_area("Options (one per line)", "\n".join(question.get("options", [])), key=f"options_{prefix}{i}")
                    edited_question['options'] = [opt.strip() for opt in options.split("\n") if opt.strip()]
                
                if "sub_questions" in question and isinstance(question["sub_questions"], list):
                    edited_question["sub_questions"] = display_questionnaire(question["sub_questions"], prefix=f"{prefix}{i+1}.")
            
            edited_questions.append(edited_question)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Add Sub-Question", key=f"add_sub_{prefix}{i}"):
                    new_sub = {"question": "New Sub-Question", "type": "text", "instructions": ""}
                    if "sub_questions" not in edited_question:
                        edited_question["sub_questions"] = []
                    edited_question["sub_questions"].append(new_sub)
            with col2:
                if st.button(f"Remove Question", key=f"remove_{prefix}{i}"):
                    return [q for j, q in enumerate(edited_questions) if j != i]
        
        except Exception as e:
            st.error(f"Error processing question {prefix}{i+1}: {str(e)}")
            logger.exception(f"Error in display_questionnaire for question {prefix}{i+1}")
    
    if st.button(f"Add New Question", key=f"add_new_{prefix}"):
        edited_questions.append({
            "question": "New Question",
            "type": "text",
            "instructions": ""
        })
    
    return edited_questions

def generate_report(questions, documents, progress_bar):
    context = "\n".join([f"Title: {doc['title']}\n{doc['text']}" for doc in documents])
    
    def process_question(question):
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
            
            response = chat_completion(messages)
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

    report = []
    for i, question in enumerate(questions):
        report.append(process_question(question))
        progress_bar.progress((i + 1) / len(questions))

    logger.info(f"Report generation complete. Total questions processed: {len(report)}")
    return report
