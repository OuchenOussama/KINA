from langchain_openai import ChatOpenAI
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestionAnsweringLayer:
    """Layer for generating natural language answers using an LLM."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=os.getenv("QA_MODEL"),
            temperature=0,
            max_tokens=512,
            openai_api_base=os.getenv("OPENROUTER_API_URL"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def generate_answer(self, query: str, user_profile: dict, neo4j_results: list, hybrid_results: list, lang : str) -> str:
        """Generate a natural language answer based on combined results."""
        try:
            context = f"Neo4j Results: {json.dumps(neo4j_results, indent=2)}\nHybrid Model Results: {json.dumps(hybrid_results, indent=2)}"
            qa_prompt = f"""
You are a medical chatbot answering a medication-related question.
Query: {query}
User Profile: {user_profile}
Context: {context}
Provide a natural language answer summarizing the results, considering the user profile (e.g., age, pregnancy, allergies).
If the user's query is irrelevant or is not related to apothecary, do not include context and just ask for clarification.
Language of your answer should be : {lang}.
"""
            response = self.llm.invoke(qa_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer due to an error."