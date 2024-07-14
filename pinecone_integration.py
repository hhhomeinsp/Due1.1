import pinecone
from pinecone import Pinecone, PodSpec
import logging
import uuid
import json
from utils import get_secret

logger = logging.getLogger(__name__)

def initialize_pinecone():
    pinecone_api_key = get_secret("PINECONE_API_KEY")
    pinecone_environment = get_secret("PINECONE_ENVIRONMENT")
    index_name = get_secret("PINECONE_INDEX_NAME")

    if not pinecone_api_key:
        logger.error("Pinecone API key is not set.")
        return None

    try:
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            logger.warning(f"Index '{index_name}' does not exist. Attempting to create new index.")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # Ensure this matches your OpenAI embedding size
                    metric='cosine',
                    spec=PodSpec(
                        environment=pinecone_environment,
                        pod_type='p1.x1'  # Adjust this based on your specific pod type
                    )
                )
                logger.info(f"Created new Pinecone index: {index_name}")
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {str(e)}")
                return None
        
        # Connect to the index
        index = pc.Index(index_name)
        return index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        return None

class PineconeConnection:
    def __init__(self, index):
        self.index = index

    def test_connection(self):
        try:
            self.index.describe_index_stats()
            return True
        except Exception as e:
            st.error(f"Connection test failed: {str(e)}")
            return False

    def add_document(self, title, text, embedding):
        try:
            id = str(uuid.uuid4())
            self.index.upsert(vectors=[(id, embedding, {"title": title, "text": text, "type": "document"})])
            return id
        except Exception as e:
            st.error(f"Error adding document: {str(e)}")
            return None

    def get_all_documents(self):
        try:
            results = self.index.query(vector=[0]*1536, filter={"type": "document"}, top_k=10000, include_metadata=True)
            return [{"id": match['id'], "title": match['metadata']['title'], "text": match['metadata']['text']} 
                    for match in results['matches']]
        except Exception as e:
            st.error(f"Error getting documents: {str(e)}")
            return []

    def delete_document(self, document_id):
        try:
            self.index.delete(ids=[document_id])
            return True
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False

    def add_questionnaire(self, title, questions):
        try:
            id = str(uuid.uuid4())
            formatted_questions = self.format_questions(questions)
            self.index.upsert(vectors=[(id, [0]*1536, {
                "title": title,
                "questions": json.dumps(formatted_questions),
                "type": "questionnaire"
            })])
            logger.info(f"Saved questionnaire '{title}' with ID: {id}")
            return id
        except Exception as e:
            logger.error(f"Error adding questionnaire: {str(e)}")
            raise

    def get_all_questionnaires(self):
        try:
            if not self.test_connection():
                raise ConnectionError(f"Failed to connect to Pinecone index: {self.index_name}")

            results = self.index.query(
                vector=[0]*1536, 
                filter={"type": "questionnaire"},
                top_k=10000,
                include_metadata=True
            )
            questionnaires = []
            for match in results['matches']:
                try:
                    questionnaires.append({
                        "id": match['id'],
                        "title": match['metadata']['title'],
                        "questions": json.loads(match['metadata']['questions'])
                    })
                except KeyError as ke:
                    logger.warning(f"Malformed questionnaire data for ID {match['id']}: {ke}")
                except json.JSONDecodeError as jde:
                    logger.warning(f"Invalid JSON for questionnaire ID {match['id']}: {jde}")
            logger.info(f"Retrieved {len(questionnaires)} questionnaires from Pinecone")
            return questionnaires
        except Exception as e:
            logger.exception(f"Error retrieving questionnaires from Pinecone: {str(e)}")
            raise

    def get_questionnaire(self, questionnaire_id):
        try:
            result = self.index.fetch(ids=[questionnaire_id])
            if questionnaire_id in result['vectors']:
                vector = result['vectors'][questionnaire_id]
                return {
                    "id": questionnaire_id,
                    "title": vector['metadata']['title'],
                    "questions": json.loads(vector['metadata']['questions'])
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving questionnaire: {str(e)}")
            return None

    def delete_questionnaire(self, questionnaire_id):
        try:
            self.index.delete(ids=[questionnaire_id])
            return True
        except Exception as e:
            st.error(f"Error deleting questionnaire: {str(e)}")
            return False

    def add_report(self, title, report):
        try:
            report_id = str(uuid.uuid4())
            vector = get_embedding(json.dumps(report))  # Convert report to string for embedding
            metadata = {
                "type": "report",
                "title": title,
                "report": json.dumps(report)  # Store the entire report as a JSON string
            }
            self.index.upsert(vectors=[(report_id, vector, metadata)])
            logger.info(f"Report added successfully with ID: {report_id}")
            return report_id
        except Exception as e:
            logger.error(f"Error adding report: {str(e)}")
            return None

    def get_all_reports(self):
        try:
            results = self.index.query(vector=[0]*1536, filter={"type": "report"}, top_k=10000, include_metadata=True)
            reports = []
            for match in results['matches']:
                try:
                    reports.append({
                        "id": match['id'],
                        "title": match['metadata']['title'],
                        "report": json.loads(match['metadata']['report'])
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode report data for ID: {match['id']}")
            return reports
        except Exception as e:
            logger.error(f"Error getting reports: {str(e)}")
            return []

    def delete_report(self, report_id):
        try:
            self.index.delete(ids=[report_id])
            return True
        except Exception as e:
            st.error(f"Error deleting report: {str(e)}")
            return False

    def get_similar_documents(self, query_embedding, top_k=3):
        try:
            results = self.index.query(query_embedding, filter={"type": "document"}, top_k=top_k, include_metadata=True)
            return [(match['id'], match['metadata']['title'], match['metadata']['text'], match['score']) 
                    for match in results['matches']]
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return []

    def format_questions(self, questions):
        formatted_questions = []
        for q in questions:
            formatted_q = {
                "question": q.get("question", ""),
                "type": q.get("type", "text"),
                "instructions": q.get("instructions", ""),
            }
            if "options" in q and q["type"] == "multiple choice":
                formatted_q["options"] = q["options"]
            if "sub_questions" in q:
                formatted_q["sub_questions"] = self.format_questions(q["sub_questions"])
            formatted_questions.append(formatted_q)
        return formatted_questions
