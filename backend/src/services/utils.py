import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_json_from_response(content):
    """Robust JSON extraction that handles various response formats."""
    content = content.strip()
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content)
    content = re.sub(r'\{\{', '{', content)
    content = re.sub(r'\}\}', '}', content)
    
    extraction_methods = [
        lambda c: json.loads(c),
        lambda c: json.loads(re.search(r'\{[^{}]*"entities"[^{}]*\{[^}]*\}[^{}]*\}', c, re.DOTALL).group(0)) if re.search(r'\{[^{}]*"entities"[^{}]*\{[^}]*\}[^{}]*\}', c, re.DOTALL) else None,
        lambda c: json.loads(re.search(r'\{(?:[^{}]|{[^}]*})*\}', c).group(0)) if re.search(r'\{(?:[^{}]|{[^}]*})*\}', c) else None,
        lambda c: json.loads(re.search(r'\{.*?"entities".*?\{.*?\}.*?\}', c, re.DOTALL).group(0)) if re.search(r'\{.*?"entities".*?\{.*?\}.*?\}', c, re.DOTALL) else None,
    ]
    
    for i, method in enumerate(extraction_methods):
        try:
            result = method(content)
            if result and isinstance(result, dict):
                if 'entities' in result or 'drug_database_search' in result:
                    logger.info(f"Successfully parsed JSON using method {i+1}")
                    return result
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_candidate = content[start:end+1]
            json_candidate = re.sub(r',\s*}', '}', json_candidate)
            json_candidate = re.sub(r',\s*]', ']', json_candidate)
            result = json.loads(json_candidate)
            if isinstance(result, dict):
                logger.info("Successfully parsed JSON using aggressive method")
                return result
    except json.JSONDecodeError:
        pass
    
    logger.error("Failed to extract JSON from response")
    return None