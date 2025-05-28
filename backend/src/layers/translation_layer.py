import logging
from googletrans import Translator, LANGUAGES
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_query(query: str) -> str:
    """Translate the input query to English if it's not already in English."""
    try:
        # Initialize translator
        translator = Translator()
        
        # Detect query language
        try:
            lang = detect(query)
            logger.info(f"Detected language: {lang} ({LANGUAGES.get(lang, 'Unknown')})")
        except LangDetectException:
            logger.warning("Language detection failed, assuming English")
            return "en", query

        # If language is not English, translate to English
        if lang != 'en':
            translation = translator.translate(query, dest='en')
            logger.info(f"Translated query from {lang} to English: {translation.text}")
            return translation.text
        else:
            logger.info("Query is already in English, no translation needed")
            return lang, query

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return 'en', query