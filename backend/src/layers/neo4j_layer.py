from neo4j import GraphDatabase
import os
import logging
from typing import List, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jLayer:
    """Layer for interacting with Neo4j database."""
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j Aura."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Connected to Neo4j Aura")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def _flatten_record_to_text(self, record: Dict[str, Any]) -> str:
        """Convert a Neo4j record to structured text format for LLM consumption."""
        try:
            # Extract basic information
            brand_name = record.get('brand_name', 'Unknown')
            price = record.get('price', 'Not specified')
            
            # Start building the medication description
            text_parts = [f"Medication: {brand_name}"]
            
            # Add price information
            if price != 'Not specified' and price is not None:
                text_parts.append(f"Price: {price} Dirhams")
            
            # Add therapeutic classes
            therapeutic_classes = record.get('therapeutic_classes', [])
            if therapeutic_classes and any(tc for tc in therapeutic_classes if tc):
                clean_classes = [tc for tc in therapeutic_classes if tc and tc.strip()]
                if clean_classes:
                    text_parts.append(f"Therapeutic Class: {', '.join(clean_classes)}")
            
            # Add compositions (active ingredients)
            compositions = record.get('compositions', [])
            if compositions and any(comp for comp in compositions if comp):
                clean_compositions = [comp for comp in compositions if comp and comp.strip()]
                if clean_compositions:
                    text_parts.append(f"Active Ingredients: {', '.join(clean_compositions)}")
            
            # Add dosages
            dosages = record.get('dosages', [])
            if dosages and any(dose for dose in dosages if dose):
                clean_dosages = [dose.strip() for dose in dosages if dose and dose.strip()]
                if clean_dosages:
                    text_parts.append(f"Available Dosages: {', '.join(clean_dosages)}")
            
            # Add forms
            forms = record.get('forms', [])
            if forms and any(form for form in forms if form):
                clean_forms = [form for form in forms if form and form.strip()]
                if clean_forms:
                    text_parts.append(f"Forms: {', '.join(clean_forms)}")
            
            # Add packaging
            packaging = record.get('packaging', [])
            if packaging and any(pkg for pkg in packaging if pkg):
                clean_packaging = [pkg for pkg in packaging if pkg and pkg.strip()]
                if clean_packaging:
                    text_parts.append(f"Packaging: {', '.join(clean_packaging)}")
            
            # Add medical uses
            uses = record.get('uses', [])
            if uses and any(use for use in uses if use):
                clean_uses = [use for use in uses if use and use.strip()]
                if clean_uses:
                    text_parts.append(f"Medical Uses: {', '.join(clean_uses)}")
            
            # Add considerations
            considerations = record.get('considerations', [])
            if considerations and any(cons for cons in considerations if cons):
                clean_considerations = [cons for cons in considerations if cons and cons.strip()]
                if clean_considerations:
                    text_parts.append(f"Considerations: {', '.join(clean_considerations)}")
            
            # Add contraindications
            contraindications = record.get('contraindications', [])
            if contraindications and any(contra for contra in contraindications if contra):
                clean_contraindications = [contra for contra in contraindications if contra and contra.strip()]
                if clean_contraindications:
                    text_parts.append(f"Contraindications: {', '.join(clean_contraindications)}")
            
            return " | ".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error flattening record: {e}")
            logger.error(f"Record content: {record}")
            return f"Medication: {record.get('brand_name', 'Unknown')} (Error processing details)"

    def _format_records_for_llm(self, records: List[Dict[str, Any]]) -> str:
        """Format multiple records into a single text block for LLM context."""
        if not records:
            return "No medications found matching your criteria."
        
        formatted_records = []
        for i, record in enumerate(records, 1):
            flattened = self._flatten_record_to_text(record)
            formatted_records.append(f"{i}. {flattened}")
        
        header = f"Found {len(records)} medication(s):\n\n"
        result = header + "\n\n".join(formatted_records)
        
        # Log the final result as a single entity, not character by character
        logger.info(f"Formatted {len(records)} records for LLM context")
        logger.debug(f"LLM context preview: {result[:200]}...")  # Only log first 200 chars for preview
        
        return result

    def execute_query(self, cypher_query: str, return_format: str = "text") -> Union[List[Dict[str, Any]], str]:
        if not self.driver:
            logger.error("No Neo4j connection available")
            return [] if return_format == "raw" else "Database connection unavailable."
        
        try:
            with self.driver.session() as session:
                logger.info(f"Executing Neo4j query: {cypher_query}")
                result = session.run(cypher_query)
                
                # Collect records explicitly
                records = []
                for record in result:
                    logger.debug(f"Raw Neo4j record: {record}")
                    record_data = record.data()
                    logger.debug(f"Processed record data: {record_data}")
                    records.append(record_data)

                num_records = len(records)
                
                logger.info(f"Neo4j query returned {num_records} records")
                
                for i, record in enumerate(records[:3], 1):
                    logger.info(f"Record {i} keys: {list(record.keys())}")
                
                if return_format == "text":
                    formatted_result = self._format_records_for_llm(records)
                    return formatted_result, num_records
                else:
                    return records, num_records
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            logger.error(f"Query was: {cypher_query}")
            if return_format == "text":
                return f"Error retrieving medication data: {str(e)}"
            else:
                return [], 0
                    
    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")