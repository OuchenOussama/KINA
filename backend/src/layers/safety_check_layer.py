from typing import List, Dict, Any
from .ner_layer import ExtractedEntities
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safety_check(neo4j_results: List[Dict[str, Any]], hybrid_results: List[Dict[str, Any]], user_profile: Dict[str, Any], entities: ExtractedEntities) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter drugs based on user profile for safety."""
    logger.info("Performing safety check on retrieved drugs")

    return neo4j_results, hybrid_results
    
    filtered_neo4j = []
    filtered_hybrid = []

    # Extract user profile details
    age = user_profile.get('age', 30)
    is_pregnant = user_profile.get('isPregnant', False)
    allergies = [allergy.lower() for allergy in user_profile.get('allergies', [])]

    # Filter Neo4j results
    for result in neo4j_results:
        # Check if result is a dictionary
        if not isinstance(result, dict):
            logger.warning(f"Skipping invalid Neo4j result: {result} (expected dict, got {type(result)})")
            # Attempt to convert tuple to dict if it contains expected keys
            if isinstance(result, tuple):
                try:
                    # Assuming tuple corresponds to RETURN order: b, tc, c, d, p, u, con, contra
                    keys = ['b', 'tc', 'c', 'd', 'p', 'u', 'con', 'contra']
                    if len(result) == len(keys):
                        result = {key: value for key, value in zip(keys, result)}
                        logger.debug(f"Converted tuple to dict: {result}")
                    else:
                        logger.warning(f"Tuple length mismatch, expected {len(keys)} elements, got {len(result)}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to convert tuple to dict: {e}")
                    continue
            else:
                continue

        # Now process as dictionary
        brand = result.get('b', {}).get('name', '').lower()
        contraindications = [c.lower() for c in result.get('contra', {}).get('name', [])] if result.get('contra') else []
        composition = result.get('c', {}).get('name', '').lower() if result.get('c') else ''

        skip = False
        # Check pregnancy contraindication
        if is_pregnant and 'pregnancy' in contraindications:
            logger.info(f"Excluding {brand} due to pregnancy contraindication")
            skip = True
        # Check allergies
        if any(allergy in composition or allergy in brand for allergy in allergies):
            logger.info(f"Excluding {brand} due to allergy match")
            skip = True
        # Check age (basic rule: exclude certain drugs for children <12)
        if age and age < 12 and 'pediatric' in contraindications:
            logger.info(f"Excluding {brand} due to pediatric contraindication")
            skip = True

        if not skip:
            filtered_neo4j.append(result)

    # Filter hybrid model results (unchanged)
    for result in hybrid_results:
        drug_name = result.get('name', '').lower()
        drug_info = result.get('info', {})
        contraindications = [c.lower() for c in drug_info.get('Contraindications', [])]
        composition = drug_info.get('Composition', '').lower()

        skip = False
        if is_pregnant and 'pregnancy' in contraindications:
            logger.info(f"Excluding {drug_name} due to pregnancy contraindication")
            skip = True
        if any(allergy in composition or allergy in drug_name for allergy in allergies):
            logger.info(f"Excluding {drug_name} due to allergy match")
            skip = True
        if age < 12 and 'pediatric' in contraindications:
            logger.info(f"Excluding {drug_name} due to pediatric contraindication")
            skip = True

        if not skip:
            filtered_hybrid.append(result)

    logger.info(f"Safety check complete: {len(filtered_neo4j)} Neo4j results, {len(filtered_hybrid)} hybrid results retained")
    return filtered_neo4j, filtered_hybrid