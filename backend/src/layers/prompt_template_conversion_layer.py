import logging
from typing import Optional
from .ner_layer import ExtractedEntities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_template(entities: ExtractedEntities) -> Optional[str]:
    """
    Convert extracted entities into a combined query string that mirrors the knowledge graph structure.
    The query is formatted as 'BrandName:brand used for ... packaged as ...' for semantic search.
    Returns a single string optimized for find_drugs_smart.
    """
    try:
        query_parts = []

        # BrandName
        if entities.brand:
            for brand in entities.brand:
                query_parts.append(f"BrandName:{brand}")

        # Composition
        if entities.composition:
            for comp in entities.composition:
                query_parts.append(f"contains active ingredient {comp}")

        # TherapeuticClass
        if entities.therapeutic_class:
            for tc in entities.therapeutic_class:
                query_parts.append(f"in therapeutic class : {tc}")

        # Uses
        if entities.use:
            for use in entities.use:
                query_parts.append(f"indicated for {use}")

        # Contraindications
        if entities.contraindication:
            for contraindication in entities.contraindication:
                query_parts.append(f"contraindicated in {contraindication}")

        # Considerations
        if entities.consideration:
            for consideration in entities.consideration:
                query_parts.append(f"clinical consideration {consideration}")

        # Form
        if entities.form:
            for form in entities.form:
                query_parts.append(f"pharmaceutical form of {form}")

        # Dosage (value and unit)
        if entities.dosage_value and entities.dosage_unit:
            dosage_str = f"dosage strength of {entities.dosage_value[0]} {entities.dosage_unit[0]}"
            query_parts.append(dosage_str)

        # Packaging
        if entities.packaging:
            for pkg in entities.packaging:
                query_parts.append(f"packaged as {pkg}")

        # Remove duplicates while preserving order
        query_parts = list(dict.fromkeys(query_parts))

        # Combine into a single query string
        combined_prompt = ' '.join(query_parts) if query_parts else ''

        logger.info(f"Generated combined query: {combined_prompt}")
        return combined_prompt

    except Exception as e:
        logger.error(f"Error in query conversion: {e}")
        # Fallback to the use entity or empty string
        return entities.use[0] if entities.use else ''