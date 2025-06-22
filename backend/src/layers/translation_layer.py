import logging
from typing import Tuple
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def translate_query(query: str, target_lang: str = "en") -> Tuple[str, str]:
    """
    Translate a query to the target language.
    
    Args:
        query: The text to translate
        target_lang: Target language code (default: "en")
        
    Returns:
        Tuple of (detected_language, translated_text)
        If translation fails, returns (detected_language, original_query)
    """
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return "unknown", query
    
    # For single words, assume English (language detection is unreliable)
    query_stripped = query.strip()
    if len(query_stripped.split()) == 1:
        logger.info("Single word query detected, assuming English")
        return "en", query_stripped
    
    # Detect source language
    try:
        detected_lang = detect(query_stripped)
        logger.info(f"Detected language: {detected_lang}")
    except LangDetectException:
        logger.warning("Language detection failed, assuming English")
        detected_lang = "en"
    
    # Skip translation if already in target language
    if detected_lang == target_lang:
        logger.info(f"Query already in target language ({target_lang})")
        return detected_lang, query
    
    # Perform translation
    try:
        translator = GoogleTranslator(source=detected_lang, target=target_lang)
        translated_text = translator.translate(query_stripped)
        
        if not translated_text:
            logger.warning("Translation returned empty result")
            return detected_lang, query
            
        logger.info(f"Successfully translated from {detected_lang} to {target_lang}")
        return detected_lang, translated_text
        
    except Exception as e:
        logger.error(f"Translation failed: {type(e).__name__}: {e}")
        return detected_lang, query
