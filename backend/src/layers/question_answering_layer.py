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
            context = f"Database Results: {json.dumps(neo4j_results, indent=2)}\nRetrieval Model Results: {json.dumps(hybrid_results, indent=2)}"
            qa_prompt = f"""
            You are a **professional pharmaceutical assistant** helping licensed pharmacists respond to medication-related questions.

Your job is to provide accurate, context-specific information using only the data provided below.

---

## INPUT:
- User Query: {query}
- Patient Risk Flags: {flags}
- Drug Information Context: {context}

---

## INSTRUCTIONS:

1. If the question is unrelated to medications, prescriptions, or pharmacy care, ask politely for clarification. Do not refer to the context.

2. Respond as a human pharmacist. Do not mention AI, data retrieval, or backend processes.

3. Base your answer strictly on the information in the provided context. Do not guess or invent information.

4. Mention only drugs that are directly relevant to the query.

5. For each drug you include, follow this exact format (one drug per line):

   *DRUGNAME* — Primary use. Dosage info if available. Price. Key warnings or contraindications.

   Example:
   *COLPRONE* — Used for cramps and dysmenorrhea. 100 mg, 2–3 times per day. 23.50 MAD. Contraindicated in pregnancy and breastfeeding.

6. Adapt your answer to the patient's risk flags. For example:
   - If pregnant: warn about any pregnancy-related contraindications.
   - If kidney issues: note any renal precautions or dose adjustments.
   - If allergies: avoid drugs with known hypersensitivity risks.

7. Do not repeat the query. Begin with the answer directly.

8. Write in clear, professional paragraphs. Use markdown bullet points (starting with *) only when listing multiple related items. Do not use bullet symbols (•) in your text.

9. Format your final answer as **Markdown**.

---

## REMINDER:
Do not hallucinate. Use only the provided context. If the query is too vague, ask for clarification like a human pharmacist would.
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