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
                        'pediatric': {'min_age': 16, 'risk': RiskLevel.CRITICAL, 'details': 'Risk of Reye’s syndrome'},
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
    neo4j_results: List[Dict[str, Any]],
    hybrid_results: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    entities: ExtractedEntities) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[SafetyFlag]]:
    """Perform safety checks on Neo4j and hybrid results using predefined rules."""
    checker = EnhancedSafetyChecker()
    filtered_neo4j = []
    filtered_hybrid = []
    all_flags = []

    # Extract user data
    user_data = _extract_user_data(user_profile)

    # Process Neo4j results
    for result in neo4j_results:
        drug_info = _extract_drug_info_neo4j(result)
        flags = _check_drug_safety(drug_info, user_data, checker)
        all_flags.extend(flags)
        if _is_safe_to_recommend(flags):
            filtered_neo4j.append(drug_info)

    # Process hybrid results
    for result in hybrid_results:
        drug_info = _extract_drug_info_hybrid(result)
        flags = _check_drug_safety(drug_info, user_data, checker)
        all_flags.extend(flags)
        if _is_safe_to_recommend(flags):
            filtered_hybrid.append(drug_info)

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
    """Extract drug information from Neo4j result."""
    try:
        brand_name = str(result.get('b', {}).get('name', '')).lower()
        return {
            'name': brand_name if brand_name else 'unknown',
            'source': 'neo4j'
        }
    except Exception as e:
        logger.error(f"Error extracting Neo4j drug info: {e}")
        return {'name': 'unknown', 'source': 'neo4j'}

def _extract_drug_info_hybrid(result: Any) -> Dict[str, Any]:
    """Extract drug information from hybrid result (tuple format)."""
    try:
        drug_name = str(result[0]).lower() if isinstance(result, tuple) and len(result) > 0 else 'unknown'
        return {
            'name': drug_name,
            'source': 'hybrid'
        }
    except Exception as e:
        logger.error(f"Error extracting hybrid drug info: {e}")
        return {'name': 'unknown', 'source': 'hybrid'}

def _check_drug_safety(
    drug_info: Dict[str, Any],
    user_data: Dict[str, Any],
    checker: EnhancedSafetyChecker
) -> List[SafetyFlag]:
    """Check drug safety based on predefined rules."""
    drug_name = drug_info['name'].lower()
    flags = []

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
            if 'drug_interactions' in rules:
                for med in user_data['current_medications']:
                    if med in rules['drug_interactions']:
                        interaction = rules['drug_interactions'][med]
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

    # Flag unknown drugs
    if not flags and drug_name != 'unknown':
        found = any(drug_name == rule_drug or drug_name in checker.safety_rules['drugs'].get(rule_drug, {}).get('aliases', []) 
                    for rule_drug in checker.safety_rules['drugs'])
        if not found:
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