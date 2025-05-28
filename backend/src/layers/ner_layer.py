import os
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import logging
from services.utils import extract_json_from_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtractedEntities(BaseModel):
    """Named entities extracted from user queries about medications."""
    
    brand: Optional[List[str]] = Field(None, description="Brand names of medications (e.g., Tylenol, Advil, Aspirin)")
    composition: Optional[List[str]] = Field(None, description="Active ingredients/compositions (e.g., paracetamol, ibuprofen, amoxicillin)")
    consideration: Optional[List[str]] = Field(None, description="Medical considerations or warnings (e.g., 'take with food', 'avoid alcohol')")
    contraindication: Optional[List[str]] = Field(None, description="Contraindications or conditions where medication should be avoided")
    dosage_value: Optional[List[float]] = Field(None, description="Dosage values as numbers (e.g., 500, 1000, 2.5)")
    dosage_unit: Optional[List[str]] = Field(None, description="Dosage units (e.g., mg, ml, g, units)")
    packaging: Optional[List[str]] = Field(None, description="Packaging types (e.g., box, bottle, blister pack, vial)")
    form: Optional[List[str]] = Field(None, description="Medication forms (e.g., tablet, syrup, injection, capsule, cream)")
    therapeutic_class: Optional[List[str]] = Field(None, description="Therapeutic classes (e.g., antibiotics, analgesics, antihypertensives)")
    use: Optional[List[str]] = Field(None, description="Medical uses or indications (e.g., 'for pain relief', 'to treat infection')")
    price: Optional[List[float]] = Field(None, description="Specific price values mentioned (e.g., 'costs 25 dollars')")
    price_min: Optional[float] = Field(None, description="Minimum price mentioned (e.g., 'more than 10 dollars')")
    price_max: Optional[float] = Field(None, description="Maximum price mentioned (e.g., 'cheaper than 20 dollars')")
    sort_preference: Optional[str] = Field(None, description="Sorting preference (e.g., 'cheapest', 'most expensive', 'alphabetical')")
    limit: Optional[int] = Field(None, description="Number of results requested (e.g., 'show me 5 medications')")

class NamedEntityRecognizer:
    """Layer for extracting named entities from medication queries."""
    
    def __init__(self, llm):
        self.llm = llm
        self.ner_analyzer = self._create_ner_chain()
    
    def _create_ner_chain(self):
        """Create the NER chain with few-shot examples."""
        ner_examples = [
            {
                "question": "Find me all medications containing paracetamol with a dosage greater than 500mg",
                "answer": """{
    "entities": {
        "brand": null,
        "composition": ["paracetamol"],
        "consideration": null,
        "contraindication": null,
        "dosage_value": [500.0],
        "dosage_unit": ["mg"],
        "packaging": null,
        "form": null,
        "therapeutic_class": null,
        "use": null,
        "price": null,
        "price_min": 500.0,
        "price_max": null,
        "sort_preference": null,
        "limit": null
    }
}"""
            },
            {
                "question": "What are the cheapest antibiotics available in syrup form?",
                "answer": """{
    "entities": {
        "brand": null,
        "composition": null,
        "consideration": null,
        "contraindication": null,
        "dosage_value": null,
        "dosage_unit": null,
        "packaging": null,
        "form": ["syrup"],
        "therapeutic_class": ["antibiotics"],
        "use": null,
        "price": null,
        "price_min": null,
        "price_max": null,
        "sort_preference": "cheapest",
        "limit": null
    }
}"""
            },
            {
                "question": "Show me the 10 most expensive erectile dysfunction tablets",
                "answer": """{
    "entities": {
        "brand": null,
        "composition": null,
        "consideration": null,
        "contraindication": null,
        "dosage_value": null,
        "dosage_unit": null,
        "packaging": null,
        "form": ["tablet"],
        "therapeutic_class": ["erectile dysfunction"],
        "use": null,
        "price": null,
        "price_min": null,
        "price_max": null,
        "sort_preference": "most expensive",
        "limit": 10
    }
}"""
            },
            {
                "question": "I'm looking for Tylenol in bottle form under 15 dollars",
                "answer": """{
    "entities": {
        "brand": ["Tylenol"],
        "composition": null,
        "consideration": null,
        "contraindication": null,
        "dosage_value": null,
        "dosage_unit": null,
        "packaging": ["bottle"],
        "form": null,
        "therapeutic_class": null,
        "use": null,
        "price": null,
        "price_min": null,
        "price_max": 15.0,
        "sort_preference": null,
        "limit": null
    }
}"""
            },
            {
                "question": "Find insulin injections that should not be used with alcohol",
                "answer": """{
    "entities": {
        "brand": null,
        "composition": ["insulin"],
        "consideration": null,
        "contraindication": ["alcohol"],
        "dosage_value": null,
        "dosage_unit": null,
        "packaging": null,
        "form": ["injection"],
        "therapeutic_class": null,
        "use": null,
        "price": null,
        "price_min": null,
        "price_max": null,
        "sort_preference": null,
        "limit": null
    }
}"""
            },
            {
                "question": "Show me pain relief medications that cost exactly 12.50 with English instructions",
                "answer": """{
    "entities": {
        "brand": null,
        "composition": null,
        "consideration": null,
        "contraindication": null,
        "dosage_value": null,
        "dosage_unit": null,
        "packaging": null,
        "form": null,
        "therapeutic_class": null,
        "use": ["pain relief"],
        "price": [12.50],
        "price_min": null,
        "price_max": null,
        "sort_preference": null,
        "limit": null
    }
}"""
            }
        ]
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}")
        ])
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=ner_examples,
        )
        
        system_message = """You are an expert Named Entity Recognition system specialized in pharmaceutical and medical domains.

Your task is to extract the following entities from user questions about medications:

1. **Brand**: Medication brand names (e.g., Tylenol, Advil, Aspirin)
2. **Composition**: Active ingredients (e.g., paracetamol, ibuprofen, amoxicillin)
3. **Consideration**: Medical considerations or warnings
4. **Contraindication**: Conditions where medication should be avoided
5. **DosageValue**: Numeric dosage values (e.g., 500, 1000, 2.5)
6. **DosageUnit**: Units of measurement (e.g., mg, ml, g)
7. **Packaging**: Package types (e.g., box, bottle, blister pack)
8. **Form**: Medication forms (e.g., tablet, syrup, injection, capsule)
9. **TherapeuticClass**: Therapeutic categories (e.g., antibiotics, analgesics)
10. **Use**: Medical uses or indications (e.g., pain relief, infection treatment)
11. **Price**: Specific price values mentioned (e.g., costs 25 dollars)
12. **Language**: Language preferences (e.g., English instructions, Spanish label)

Additional extraction:
- **price_min**: Minimum price (e.g., "more than 10 dollars")
- **price_max**: Maximum price (e.g., "cheaper than 20 dollars")
- **sort_preference**: Sorting preference (cheapest, most expensive, etc.)
- **limit**: Number of results requested

Important guidelines:
- Only extract entities that are explicitly mentioned or clearly implied
- Return arrays for entities that can have multiple values
- Use null for entities not present in the query
- Be precise in distinguishing between brand names, compositions, and therapeutic classes
- Extract numeric values accurately for dosages and prices
- If the query is gibberish or offensive, return an object with all values as null

You MUST respond with valid JSON in this exact format:
{{
    "entities": {{
        "brand": null,
        "composition": null,
        "consideration": null,
        "contraindication": null,
        "dosage_value": null,
        "dosage_unit": null,
        "packaging": null,
        "form": null,
        "therapeutic_class": null,
        "use": null,
        "price": null,
        "price_min": null,
        "price_max": null,
        "sort_preference": null,
        "limit": null
    }}
}}
"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
        
        final_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            few_shot_prompt,
            human_message_prompt
        ])
        
        return final_prompt | self.llm
    
    def extract_entities(self, user_question: str) -> Optional[ExtractedEntities]:
        """Extract entities from user question."""
        try:
            result = self.ner_analyzer.invoke({"question": user_question})
            
            if hasattr(result, 'content'):
                content = result.content
                logger.info(f"NER Raw Response: {content}")
                
                parsed_json = extract_json_from_response(content)
                
                if parsed_json and 'entities' in parsed_json:
                    entities_dict = parsed_json['entities']
                    return ExtractedEntities(**entities_dict)
                else:
                    logger.error(f"Failed to extract entities from response: {content}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return None