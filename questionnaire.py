import streamlit as st
import logging
import json
from utils import extract_text_from_file, chat_completion

logger = logging.getLogger(__name__)

def process_questionnaire(uploaded_form):
    with st.spinner("Processing questionnaire..."):
        try:
            file_content = extract_text_from_file(uploaded_form)
            questions = extract_questions(file_content)
            st.session_state["current_questionnaire"] = {"title": uploaded_form.name, "questions": questions}
            st.sidebar.success("Questionnaire processed successfully! Go to the 'Questionnaires' tab to review and edit.")
            logger.info(f"Successfully processed questionnaire: {uploaded_form.name}")
        except Exception as e:
            st.sidebar.error(f"An error occurred while processing the questionnaire: {str(e)}")
            logger.exception("Error in questionnaire processing")

def extract_questions(file_content):
    try:
        extracted_content = extract_document_content(file_content)
        prompt = f"""
        Analyze the following form or questionnaire content and extract ALL items of information being requested, including main questions, sub-questions, and any grouped questions.
        For each item, provide:
        1. The exact question or field name as it appears in the document.
        2. The type of information requested (e.g., text, number, date, yes/no, multiple choice, file upload, etc.).
        3. Any additional instructions or context provided for the question.
        4. If it's a sub-question or part of a group, indicate the parent question or group name.
        5. Any options provided for multiple choice questions.
        Your response must be a valid JSON array of objects. Each object should represent a question or field.
        Do not include any explanatory text outside the JSON array.
        Ensure the response starts with '[' and ends with ']'.
        Document content:
        {extracted_content[:3000]}  # Limit content to first 3000 characters for brevity
        """
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in analyzing forms and questionnaires. Your response must be a valid JSON array of objects, nothing else."},
            {"role": "user", "content": prompt}
        ]
        response = chat_completion(messages)
        ai_response = response['choices'][0]['message']['content'].strip()
        logger.info(f"AI Response: {ai_response[:1000]}...")  # Keep logging for backend debugging
        
        if not ai_response:
            raise ValueError("The AI provided an empty response.")
        
        json_match = re.search(r'\[[\s\S]*\]', ai_response)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("Could not find a JSON array in the AI response.")
        
        questions = json.loads(json_str)
        structured_questions = structure_and_validate_questions(questions)
        if not structured_questions:
            raise ValueError("No valid questions were extracted from the questionnaire.")
        
        return structured_questions
    except ValueError as e:
        logger.error(str(e))
        st.error(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in extract_questions: {str(e)}")
        st.error("An unexpected error occurred while processing the questionnaire. Please try again.")
    
    return []

def structure_and_validate_questions(questions):
    if not isinstance(questions, list):
        logger.warning(f"Expected a list of questions, but got: {type(questions)}")
        return []

    structured_questions = []
    for q in questions:
        if not isinstance(q, dict):
            logger.warning(f"Skipped invalid question (not a dictionary): {q}")
            continue
        
        if all(key in q for key in ['question', 'type', 'instructions']):
            q['type'] = q['type'].lower()
            if q['type'] not in ['text', 'number', 'date', 'yes/no', 'multiple choice', 'file upload']:
                q['type'] = 'text'
                logger.warning(f"Invalid question type for '{q['question']}'. Defaulting to 'text'.")
            
            if 'parent' in q:
                parent = next((p for p in structured_questions if p['question'] == q['parent']), None)
                if parent:
                    if 'sub_questions' not in parent:
                        parent['sub_questions'] = []
                    parent['sub_questions'].append(q)
                else:
                    logger.warning(f"Parent question '{q['parent']}' not found for '{q['question']}'. Adding as main question.")
                    structured_questions.append(q)
            else:
                structured_questions.append(q)
        else:
            logger.warning(f"Skipped invalid question (missing required keys): {q}")
    
    return structured_questions

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

def display_questionnaire_tab(pinecone_connection):
    st.header("Questionnaire Management")
    
    # Display current questionnaire
    if "current_questionnaire" in st.session_state:
        st.subheader(f"Current Questionnaire: {st.session_state['current_questionnaire']['title']}")
        questions = st.session_state["current_questionnaire"].get("questions", [])
        if questions:
            # Questionnaire Editing
            st.subheader("Edit Questionnaire")
            edited_questions = display_questionnaire(questions)
            
            col1, col2 = st.columns(2)
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
                    try:
                        with st.spinner("Generating report..."):
                            documents = pinecone_connection.get_all_documents()
                            progress_bar = st.progress(0)
                            report = generate_report(edited_questions, documents, progress_bar)
                        
                        logger.info(f"Report generated. Attempting to save...")
                        report_id = pinecone_connection.add_report(f"Report for {st.session_state['current_questionnaire']['title']}", report)
                        if report_id:
                            logger.info(f"Report saved successfully with ID: {report_id}")
                            st.success(f"Report generated and saved successfully. View it in the 'Generated Reports' tab.")
                        else:
                            logger.error("Failed to save the generated report.")
                            st.error("Failed to save the generated report.")
                    except Exception as e:
                        logger.exception(f"Error generating or saving report: {str(e)}")
                        st.error(f"Error generating or saving report: {str(e)}")
        else:
            st.warning("The current questionnaire has no questions.")
    else:
        st.info("No current questionnaire to display. Please upload a questionnaire from the sidebar or load a saved one.")

    # Display saved questionnaires
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

    # Option to create a new questionnaire
    if st.button("Create New Questionnaire"):
        st.session_state["current_questionnaire"] = {
            "title": "New Questionnaire",
            "questions": []
        }
        st.experimental_rerun()
