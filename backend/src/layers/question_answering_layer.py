from .translation_layer import translate_query
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

    def generate_answer(self, query: str, flags: dict, neo4j_results: list, hybrid_results: list, lang : str) -> str:
        """Generate a natural language answer based on combined results."""
        try:
            context = f"Neo4j Results: {json.dumps(neo4j_results, indent=2)}\nHybrid Model Results: {json.dumps(hybrid_results, indent=2)}"
            qa_prompt = f"""
You are a pharmaceutical medical chatbot answering a medication-related question, helping a pharmacist in treating a user's request.
Query: {query}
Risk Flags based on User Profile: {flags}
Context: {context}
Provide a natural language answer presenting the drugs. Do not mention retrieval methods, you're a pharmacist. 
If the user's query is irrelevant or is not related to apothecary, do not include context and just ask for clarification.
Return response in the form of Markdown text, no response title tho. Emphasize names with asteriscs and no bullet points.
Give Info about each drug from the context, do not hallucinate.
"""
            response = self.llm.invoke(qa_prompt)
            if lang == 'en':
                return response.content
            else:
                _, translated_response = translate_query(response.content, lang)
                return translated_response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I couldn't generate an answer due to an error."