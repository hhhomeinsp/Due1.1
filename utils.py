import streamlit as st

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
