import logging
from spellchecker import SpellChecker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize spell checker with English dictionary
spell = SpellChecker()

# Custom medical terms to add to the dictionary
medical_terms = [
    'paracetamol', 'ibuprofen', 'amoxicillin', 'antibiotic', 'analgesic',
    'antihistamine', 'antipyretic', 'therapeutic', 'dosage', 'formulation',
    'contraindication', 'tylenol', 'advil', 'aspirin', 'insulin'
]

# Add medical terms to spell checker dictionary
spell.word_frequency.load_words(medical_terms)

def correct_spelling(query: str) -> str:
    """Correct spelling errors in the input query."""
    try:
        # Split query into words
        words = query.split()
        corrected_words = []

        for word in words:
            # Check if the word is likely misspelled
            if word.lower() not in spell and not word.isdigit():
                corrected = spell.correction(word)
                if corrected != word:
                    logger.info(f"Corrected '{word}' to '{corrected}'")
                corrected_words.append(corrected if corrected else word)
            else:
                corrected_words.append(word)

        corrected_query = ' '.join(corrected_words)
        if corrected_query != query:
            logger.info(f"Spelling corrected query: '{query}' -> '{corrected_query}'")
        else:
            logger.info("No spelling corrections needed")
        return corrected_query

    except Exception as e:
        logger.error(f"Spelling correction error: {e}")
        return query