import logging
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def translate_query(query: str, to_lang : str = "en") -> tuple[str, str]:
    """Translate the input query."""
    try:
        # Detect query language
        try:
            lang = detect(query)
        except LangDetectException:
            logger.warning("Language detection failed, assuming English")
            return 'en', query

        try:
            translator = GoogleTranslator(source=lang, target=to_lang)
            translated_text = translator.translate(query)
            
            if translated_text:
                return lang, translated_text
            else:
                return lang, query
                
        except Exception as trans_error:
            logger.error(f"Translation failed: {trans_error}")
            logger.info("Returning original query")
            return lang, query

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return 'en', query