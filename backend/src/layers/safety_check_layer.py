from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from layers.ner_layer import ExtractedEntities

# Configure logging (reduced to warnings and errors)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SafetyFlag:
    reason: str
    risk_level: RiskLevel
    category: str
    details: str = ""

class EnhancedSafetyChecker:
    """Simplified safety checker for pharmaceutical recommendations based on predefined rules."""
    
    def __init__(self):
        # Initialize the single dataset of safety rules
        self.safety_rules = self._init_safety_rules()
    
    def _init_safety_rules(self) -> Dict:
        """Initialize predefined safety rules dataset."""
        return {
            'drugs': {
                'warfarin': {
                    'aliases': ['coumadin', 'jantoven'],
                    'contraindications': {
                        'pregnancy': {'risk': RiskLevel.CRITICAL, 'details': 'Teratogenic, risk of fetal harm'},
                        'liver_disease': {'risk': RiskLevel.HIGH, 'details': 'Requires monitoring due to hepatic metabolism'},
                        'drug_interactions': {
                            'aspirin': {'risk': RiskLevel.CRITICAL, 'details': 'Increased bleeding risk'},
                            'amiodarone': {'risk': RiskLevel.CRITICAL, 'details': 'Altered metabolism'}
                        }
                    },
                    'age_restrictions': {
                        'pediatric': {'min_age': 18, 'risk': RiskLevel.CRITICAL, 'details': 'Not safe for children'},
                        'geriatric': {'risk': RiskLevel.HIGH, 'details': 'Dose adjustment required'}
                    },
                    'allergies': {
                        'warfarin': {'risk': RiskLevel.CRITICAL, 'details': 'Known allergy'}
                    }
                },
                'aspirin': {
                    'aliases': ['bayer aspirin', 'ecotrin'],
                    'contraindications': {
                        'pregnancy': {'risk': RiskLevel.HIGH, 'details': 'Use only for preeclampsia prevention, max 150mg/day'},
                        'kidney_disease': {'risk': RiskLevel.CRITICAL, 'details': 'Risk of renal damage'},
                        'drug_interactions': {
                            'warfarin': {'risk': RiskLevel.CRITICAL, 'details': 'Increased bleeding risk'}
                        }
                    },
                    'age_restrictions': {
                        'pediatric': {'min_age': 16, 'risk': RiskLevel.CRITICAL, 'details': 'Risk of Reye\'s syndrome'},
                        'geriatric': {'risk': RiskLevel.MEDIUM, 'details': 'Monitor for GI bleeding'}
                    },
                    'allergies': {
                        'aspirin': {'risk': RiskLevel.CRITICAL, 'details': 'Known allergy'},
                        'ns cevap': {'risk': RiskLevel.HIGH, 'details': 'Cross-reactivity risk'}
                    },
                    'dosage_limits': {
                        'adult': {'max_dose': 4000, 'unit': 'mg/day'},
                        'pediatric': {'max_dose': 0, 'unit': 'mg/day', 'details': 'Contraindicated in children'}
                    }
                },
                'amoxicillin': {
                    'aliases': ['amoxil'],
                    'contraindications': {
                        'allergies': {
                            'penicillin': {'risk': RiskLevel.CRITICAL, 'details': 'Cross-reactivity with penicillin allergy'}
                        }
                    },
                    'age_restrictions': {
                        'pediatric': {'min_age': 0, 'risk': RiskLevel.LOW, 'details': 'Safe with appropriate dosing'}
                    }
                },
                'betamethasone': {
                    'aliases': ['diprolene', 'diprosalic'],
                    'contraindications': {
                        'pregnancy': {'risk': RiskLevel.HIGH, 'details': 'Topical use may be safe with medical supervision'},
                        'breastfeeding': {'risk': RiskLevel.MEDIUM, 'details': 'Minimal systemic absorption, monitor infant'}
                    },
                    'age_restrictions': {
                        'pediatric': {'min_age': 12, 'risk': RiskLevel.HIGH, 'details': 'Avoid prolonged use in children'},
                        'geriatric': {'risk': RiskLevel.MEDIUM, 'details': 'Monitor for skin thinning'}
                    }
                },
                'drill toux sèche adulte': {
                    'aliases': [],
                    'contraindications': {
                        'pregnancy': {'risk': RiskLevel.MEDIUM, 'details': 'Consult physician before use'},
                        'breastfeeding': {'risk': RiskLevel.MEDIUM, 'details': 'Limited data, consult physician'}
                    },
                    'age_restrictions': {
                        'pediatric': {'min_age': 18, 'risk': RiskLevel.CRITICAL, 'details': 'Not safe for children'},
                        'geriatric': {'risk': RiskLevel.LOW, 'details': 'Safe with standard dosing'}
                    }
                }
            },
            'whitelist': {
                'pregnancy': {
                    'aspirin': {'condition': 'preeclampsia prevention', 'max_dose': 150, 'unit': 'mg/day', 'risk': RiskLevel.MEDIUM, 'details': 'Safe under medical supervision'},
                    'betamethasone': {'condition': 'topical use for dermatoses', 'risk': RiskLevel.MEDIUM, 'details': 'Safe for short-term use under medical supervision'}
                }
            }
        }

def safety_check(
    neo4j_results: Any,  # Changed from List to Any to handle strings
    hybrid_results: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    entities: ExtractedEntities) -> Tuple[Any, List[Dict[str, Any]], List[SafetyFlag]]:
    """Perform safety checks on Neo4j and hybrid results using predefined rules."""
    checker = EnhancedSafetyChecker()
    all_flags = []

    # Extract user data
    user_data = _extract_user_data(user_profile)

    # Handle Neo4j results - they might be a formatted string or list
    filtered_neo4j = neo4j_results  # For now, just pass through
    if isinstance(neo4j_results, str):
        # If it's a string (formatted results), we can't easily extract individual drugs
        # So we'll just pass it through and flag it for manual review
        logger.info("Neo4j results are in string format, passing through for LLM processing")
        # We could try to parse the string to extract drug names, but it's complex
        # For now, let's add a general safety flag
        all_flags.append(SafetyFlag(
            reason="Neo4j results require manual safety review",
            risk_level=RiskLevel.LOW,
            category="review_required",
            details="Results are in text format and need LLM safety evaluation"
        ))
    elif isinstance(neo4j_results, list):
        # Process as list of dictionaries
        filtered_neo4j = []
        for result in neo4j_results:
            drug_info = _extract_drug_info_neo4j(result)
            if drug_info['name'] != 'unknown':
                flags = _check_drug_safety(drug_info, user_data, checker)
                all_flags.extend(flags)
                if _is_safe_to_recommend(flags):
                    filtered_neo4j.append(result)

    # Process hybrid results (these should be a list)
    filtered_hybrid = []
    if isinstance(hybrid_results, list):
        for result in hybrid_results:
            drug_info = _extract_drug_info_hybrid(result)
            if drug_info['name'] != 'unknown':
                flags = _check_drug_safety(drug_info, user_data, checker)
                all_flags.extend(flags)
                if _is_safe_to_recommend(flags):
                    filtered_hybrid.append(result)
    else:
        # If hybrid_results is not a list, treat it similarly to neo4j_results
        logger.warning(f"Hybrid results are not a list: {type(hybrid_results)}")
        filtered_hybrid = hybrid_results
        all_flags.append(SafetyFlag(
            reason="Hybrid results require manual safety review",
            risk_level=RiskLevel.LOW,
            category="review_required",
            details="Results are not in expected list format"
        ))

    return filtered_neo4j, filtered_hybrid, all_flags

def _extract_user_data(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate user data with defaults."""
    try:
        age = int(user_profile.get('age', 30))
    except (ValueError, TypeError):
        logger.warning(f"Invalid age value: {user_profile.get('age')}, defaulting to 30")
        age = 30

    return {
        'age': age,
        'is_pregnant': bool(user_profile.get('isPregnant', False)),
        'is_breastfeeding': bool(user_profile.get('isBreastfeeding', False)),
        'allergies': [str(allergy).lower().strip() for allergy in user_profile.get('allergies', []) if allergy],
        'medical_conditions': [str(condition).lower().strip() for condition in user_profile.get('medicalConditions', []) if condition],
        'current_medications': [str(med).lower().strip() for med in user_profile.get('currentMedications', []) if med],
        'kidney_function': str(user_profile.get('kidneyFunction', 'normal')).lower(),
        'liver_function': str(user_profile.get('liverFunction', 'normal')).lower(),
        'heart_condition': bool(user_profile.get('heartCondition', False))
    }

def _extract_drug_info_neo4j(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract drug information from Neo4j result with improved error handling."""
    try:
        # Handle different data types
        if isinstance(result, str):
            # If result is just a string, use it as the drug name
            drug_name = result.strip().lower()
            if drug_name and len(drug_name) > 1:  # Avoid single character strings
                return {'name': drug_name, 'source': 'neo4j'}
            else:
                logger.warning(f"Neo4j result is a short string: '{result}'")
                return {'name': 'unknown', 'source': 'neo4j'}
        
        if not isinstance(result, dict):
            logger.warning(f"Neo4j result is not a dict or string: {type(result)} - {result}")
            return {'name': 'unknown', 'source': 'neo4j'}
        
        # Dictionary processing
        drug_name = 'unknown'
        
        # Try different field names that might contain the drug name
        possible_fields = [
            'brand_name', 'name', 'drug_name', 'medication_name', 
            'product_name', 'title', 'label'
        ]
        
        for field in possible_fields:
            if field in result and result[field]:
                value = result[field]
                if isinstance(value, str) and value.strip():
                    drug_name = value.strip().lower()
                    logger.debug(f"Found drug name '{drug_name}' in field '{field}'")
                    break
        
        # If still no name found, try nested structures
        if drug_name == 'unknown':
            # Look for nested objects (like 'b' containing brand data)
            for key, value in result.items():
                if isinstance(value, dict):
                    for nested_field in possible_fields:
                        if nested_field in value and value[nested_field]:
                            nested_value = value[nested_field]
                            if isinstance(nested_value, str) and nested_value.strip():
                                drug_name = nested_value.strip().lower()
                                logger.debug(f"Found drug name '{drug_name}' in nested field '{key}.{nested_field}'")
                                break
                    if drug_name != 'unknown':
                        break
        
        # Final fallback: look for any string value that might be a drug name
        if drug_name == 'unknown':
            for key, value in result.items():
                if isinstance(value, str) and value.strip() and len(value.strip()) > 2:
                    # Avoid very short strings or single characters
                    potential_name = value.strip().lower()
                    if not potential_name.isdigit():  # Avoid numeric values
                        drug_name = potential_name
                        logger.debug(f"Using fallback drug name '{drug_name}' from field '{key}'")
                        break
        
        return {'name': drug_name, 'source': 'neo4j'}
        
    except Exception as e:
        logger.error(f"Error extracting Neo4j drug info: {e}")
        logger.error(f"Result type: {type(result)}")
        logger.error(f"Result content: {result}")
        return {'name': 'unknown', 'source': 'neo4j'}

def _extract_drug_info_hybrid(result: Any) -> Dict[str, Any]:
    """Extract drug information from hybrid result with improved error handling."""
    try:
        drug_name = 'unknown'
        
        # Case 1: String format (most common)
        if isinstance(result, str):
            drug_name = result.strip().lower()
            if drug_name and len(drug_name) > 1:
                return {'name': drug_name, 'source': 'hybrid'}
        
        # Case 2: Tuple format (result[0] contains drug name)
        elif isinstance(result, tuple) and len(result) > 0:
            if isinstance(result[0], str):
                drug_name = result[0].strip().lower()
        
        # Case 3: List format (take first element)
        elif isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], str):
                drug_name = result[0].strip().lower()
        
        # Case 4: Dictionary format
        elif isinstance(result, dict):
            # Try common field names for drug names
            possible_fields = [
                'brand_name', 'name', 'drug_name', 'medication_name',
                'product_name', 'title', 'label'
            ]
            
            for field in possible_fields:
                if field in result and result[field]:
                    value = result[field]
                    if isinstance(value, str) and value.strip():
                        drug_name = value.strip().lower()
                        break
        
        # Validate the extracted name
        if drug_name and drug_name != 'unknown' and len(drug_name) > 1 and not drug_name.isdigit():
            return {'name': drug_name, 'source': 'hybrid'}
        else:
            logger.warning(f"Could not extract valid drug name from hybrid result: {type(result)} - {result}")
            return {'name': 'unknown', 'source': 'hybrid'}
        
    except Exception as e:
        logger.error(f"Error extracting hybrid drug info: {e}")
        logger.error(f"Result type: {type(result)}")
        logger.error(f"Result content: {result}")
        return {'name': 'unknown', 'source': 'hybrid'}

def _check_drug_safety(
    drug_info: Dict[str, Any],
    user_data: Dict[str, Any],
    checker: EnhancedSafetyChecker
) -> List[SafetyFlag]:
    """Check drug safety based on predefined rules."""
    drug_name = drug_info['name'].lower()
    flags = []

    # Skip processing for unknown drugs
    if drug_name == 'unknown':
        return flags

    # Check if drug is whitelisted
    if user_data['is_pregnant'] and drug_name in checker.safety_rules.get('whitelist', {}).get('pregnancy', {}):
        whitelist_info = checker.safety_rules['whitelist']['pregnancy'][drug_name]
        flags.append(SafetyFlag(
            reason=f"Whitelisted drug: {drug_name}",
            risk_level=whitelist_info['risk'],
            category="whitelist",
            details=whitelist_info['details']
        ))
        return flags  # Skip other checks for whitelisted drugs

    # Find the drug in safety rules
    for rule_drug, rules in checker.safety_rules['drugs'].items():
        if drug_name == rule_drug or drug_name in rules.get('aliases', []):
            # Pregnancy check
            if user_data['is_pregnant'] and 'pregnancy' in rules.get('contraindications', {}):
                contra = rules['contraindications']['pregnancy']
                flags.append(SafetyFlag(
                    reason=f"Pregnancy contraindication: {drug_name}",
                    risk_level=contra['risk'],
                    category="pregnancy",
                    details=contra['details']
                ))

            # Breastfeeding check
            if user_data['is_breastfeeding'] and 'breastfeeding' in rules.get('contraindications', {}):
                contra = rules['contraindications']['breastfeeding']
                flags.append(SafetyFlag(
                    reason=f"Breastfeeding contraindication: {drug_name}",
                    risk_level=contra['risk'],
                    category="breastfeeding",
                    details=contra['details']
                ))

            # Age restrictions
            if 'age_restrictions' in rules:
                age_rules = rules['age_restrictions']
                if user_data['age'] < 18 and 'pediatric' in age_rules:
                    ped_rule = age_rules['pediatric']
                    if user_data['age'] < ped_rule.get('min_age', 0):
                        flags.append(SafetyFlag(
                            reason=f"Pediatric restriction: {drug_name}",
                            risk_level=ped_rule['risk'],
                            category="pediatric",
                            details=ped_rule['details']
                        ))
                if user_data['age'] > 65 and 'geriatric' in age_rules:
                    flags.append(SafetyFlag(
                        reason=f"Geriatric consideration: {drug_name}",
                        risk_level=age_rules['geriatric']['risk'],
                        category="geriatric",
                        details=age_rules['geriatric']['details']
                    ))

            # Allergy check
            if 'allergies' in rules:
                for allergy in user_data['allergies']:
                    if allergy in rules['allergies']:
                        allergy_rule = rules['allergies'][allergy]
                        flags.append(SafetyFlag(
                            reason=f"Allergy risk: {drug_name} with {allergy}",
                            risk_level=allergy_rule['risk'],
                            category="allergy",
                            details=allergy_rule['details']
                        ))

            # Medical condition check
            for condition in user_data['medical_conditions']:
                if condition in rules.get('contraindications', {}):
                    contra = rules['contraindications'][condition]
                    flags.append(SafetyFlag(
                        reason=f"Contraindication with {condition}: {drug_name}",
                        risk_level=contra['risk'],
                        category="medical_condition",
                        details=contra['details']
                    ))

            # Drug interactions
            if 'drug_interactions' in rules.get('contraindications', {}):
                for med in user_data['current_medications']:
                    if med in rules['contraindications']['drug_interactions']:
                        interaction = rules['contraindications']['drug_interactions'][med]
                        flags.append(SafetyFlag(
                            reason=f"Drug interaction: {drug_name} with {med}",
                            risk_level=interaction['risk'],
                            category="drug_interaction",
                            details=interaction['details']
                        ))

            # Dosage limits
            if 'dosage_limits' in rules:
                dosage = rules['dosage_limits']
                if user_data['age'] < 18 and 'pediatric' in dosage and dosage['pediatric']['max_dose'] == 0:
                    flags.append(SafetyFlag(
                        reason=f"No safe pediatric dose: {drug_name}",
                        risk_level=RiskLevel.CRITICAL,
                        category="dosage",
                        details=dosage['pediatric']['details']
                    ))

            # If we found the drug in our rules, don't flag it as unknown
            return flags

    # Flag unknown drugs (only if not found in our safety rules)
    flags.append(SafetyFlag(
        reason=f"Unknown drug: {drug_name}",
        risk_level=RiskLevel.MEDIUM,
        category="unknown",
        details="Drug not found in safety database, consult physician"
    ))

    return flags

def _is_safe_to_recommend(safety_flags: List[SafetyFlag]) -> bool:
    """Only eliminate drugs with CRITICAL flags."""
    return not any(flag.risk_level == RiskLevel.CRITICAL for flag in safety_flags)