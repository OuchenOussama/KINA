import logging
from typing import Optional
from .ner_layer import ExtractedEntities

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Medical synonyms aligned with knowledge graph node types
medical_synonyms = {
    'acne': ['pimples', 'zits', 'skin breakout'],
    'allergy': ['allergic reaction', 'hay fever'],
    'antibiotic': ['antibacterial', 'antimicrobial'],
    'pain': ['ache', 'discomfort', 'soreness'],
    'tablet': ['pill', 'capsule'],
    'syrup': ['liquid', 'solution'],
    'injection': ['shot', 'vial'],
    'paracetamol': ['acetaminophen'],
    'ibuprofen': ['advil', 'motrin'],
    'insulin': ['humulin', 'lantus']
}

def convert_to_template(entities: ExtractedEntities) -> Optional[str]:
    """
    Convert extracted entities into a combined query string that mirrors the knowledge graph structure.
    The query is formatted as 'BrandName:brand used for ... packaged as ...' for semantic search.
    Returns a single string optimized for find_drugs_smart.
    """
    try:
        query_parts = []

        # Helper function to add synonyms
        def add_synonyms(term, entity_type):
            synonyms = []
            for main_term, syn_list in medical_synonyms.items():
                if term.lower() == main_term or term.lower() in syn_list:
                    synonyms.extend([main_term] + syn_list)
            return list(dict.fromkeys(synonyms))  # Remove duplicates

        # BrandName
        if entities.brand:
            for brand in entities.brand:
                query_parts.append(f"BrandName:{brand}")
                query_parts.extend(f"BrandName:{syn}" for syn in add_synonyms(brand, 'brand'))

        # Composition
        if entities.composition:
            for comp in entities.composition:
                query_parts.append(f"Composition:{comp}")
                query_parts.extend(f"Composition:{syn}" for syn in add_synonyms(comp, 'composition'))

        # TherapeuticClass
        if entities.therapeutic_class:
            for tc in entities.therapeutic_class:
                query_parts.append(f"TherapeuticClass:{tc}")
                query_parts.extend(f"TherapeuticClass:{syn}" for syn in add_synonyms(tc, 'therapeutic_class'))

        # Uses
        if entities.use:
            for use in entities.use:
                query_parts.append(f"used for {use}")
                query_parts.extend(f"used for {syn}" for syn in add_synonyms(use, 'use'))

        # Form
        if entities.form:
            for form in entities.form:
                query_parts.append(f"Form:{form}")
                query_parts.extend(f"Form:{syn}" for syn in add_synonyms(form, 'form'))

        # Dosage (value and unit)
        if entities.dosage_value and entities.dosage_unit:
            dosage_str = f"Dosage:{entities.dosage_value[0]} {entities.dosage_unit[0]}"
            query_parts.append(dosage_str)

        # Packaging
        if entities.packaging:
            for pkg in entities.packaging:
                query_parts.append(f"packaged as {pkg}")
                query_parts.extend(f"packaged as {syn}" for syn in add_synonyms(pkg, 'packaging'))

        # Price range
        if entities.price_min or entities.price_max:
            price_str = f"price between {entities.price_min or 'any'} and {entities.price_max or 'any'}"
            query_parts.append(price_str)

        # Sort preference (included as a keyword for context)
        if entities.sort_preference:
            query_parts.append(f"sort by {entities.sort_preference}")

        # Limit (included as a keyword for context)
        if entities.limit:
            query_parts.append(f"limit {entities.limit}")

        # Remove duplicates while preserving order
        query_parts = list(dict.fromkeys(query_parts))

        # Combine into a single query string
        combined_query = ' '.join(query_parts) if query_parts else ''

        logger.info(f"Generated combined query: {combined_query}")
        return combined_query

    except Exception as e:
        logger.error(f"Error in query conversion: {e}")
        # Fallback to the use entity or empty string
        return entities.use[0] if entities.use else ''