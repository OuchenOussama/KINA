import logging
from spellchecker import SpellChecker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize spell checker with English dictionary
spell = SpellChecker()

medical_terms = [
    'paracetamol', 'ibuprofen', 'amoxicillin', 'antibiotic', 'analgesic',
    'antihistamine', 'antipyretic', 'therapeutic', 'dosage', 'formulation',
    'contraindication', 'tylenol', 'advil', 'aspirin', 'insulin',
    'diabetes', 'hypertension', 'asthma', 'bronchitis', 'pneumonia',
    'migraine', 'arthritis', 'eczema', 'psoriasis', 'gastritis',
    'epilepsy', 'anemia', 'hypothyroidism', 'hyperlipidemia', 'osteoporosis',
    'fever', 'cough', 'nausea', 'fatigue', 'dyspnea', 'pruritus',
    'edema', 'vertigo', 'anorexia', 'myalgia',
    'antidepressant', 'antiviral', 'antifungal', 'anticoagulant', 'diuretic',
    'beta-blocker', 'statin', 'bronchodilator', 'corticosteroid', 'immunosuppressant',
    'metformin', 'lisinopril', 'atorvastatin', 'levothyroxine', 'omeprazole',
    'salbutamol', 'fluoxetine', 'prednisone', 'warfarin', 'cetirizine',
    'high blood pressure', 'elevated blood pressure', 'bronchial asthma',
    'rheumatoid arthritis', 'joint inflammation', 'skin rash', 'itchy skin',
    'stomach inflammation', 'seizure disorder', 'low blood sugar', 'thyroid deficiency'
]

# Add medical terms to spell checker dictionary
spell.word_frequency.load_words(medical_terms)

# Dictionary to normalize derogatory/slang terms to formal medical terms
slang_to_formal = {
    'dick': 'penis',
    'cock': 'penis',
    'prick': 'penis',
    'schlong': 'penis',
    'dong': 'penis',
    'wiener': 'penis',
    'pussy': 'vagina',
    'cunt': 'vagina',
    'twat': 'vagina',
    'snatch': 'vagina',
    'cooze': 'vagina',
    'ass': 'anus',
    'asshole': 'anus',
    'butthole': 'anus',
    'shit': 'feces',
    'crap': 'feces',
    'poop': 'feces',
    'turd': 'feces',
    'piss': 'urine',
    'pee': 'urine',
    'whiz': 'urine',
    'cum': 'semen',
    'jizz': 'semen',
    'spunk': 'semen',
    'tits': 'breasts',
    'boobs': 'breasts',
    'jugs': 'breasts',
    'knockers': 'breasts',
    'rack': 'breasts',
    'balls': 'testicles',
    'nuts': 'testicles',
    'bollocks': 'testicles',
    'stones': 'testicles',
    'impotence': 'erectile dysfunction',
    'ED': 'erectile dysfunction',
    'limp dick': 'erectile dysfunction',
    'clap': 'gonorrhea',
    'crabs': 'pubic lice',
    'junk': 'genitalia',
    'down there': 'genital area',
    'cooze': 'vagina',
    'std': 'sexually transmitted disease',
    'STI': 'sexually transmitted infection',
    'herpes': 'herpes simplex virus',
    'chlamydia': 'chlamydia infection',
    'syph': 'syphilis',
    'boner': 'erection',
    'hard-on': 'erection',
    'period': 'menstruation',
    'time of the month': 'menstruation',
    'on the rag': 'menstruation',
    'cramps': 'menstrual pain',
    'fucked up': 'dysfunctional',
    'messed up': 'dysfunctional',
    'screwed up': 'dysfunctional',
    'shits': 'diarrhea',
    'toot': 'flatulence',
    'blue balls': 'testicular discomfort',
    'wet dream': 'nocturnal emission',
    'jerk off': 'masturbation',
    'wank': 'masturbation',
    'jack off': 'masturbation',
    'beat off': 'masturbation',
    'rub one out': 'masturbation',
    'blowjob': 'fellatio',
    'BJ': 'fellatio',
    'go down': 'oral sex',
    'eat out': 'cunnilingus',
    'rimjob': 'anilingus',
    'buttfuck': 'anal intercourse',
    'sodomy': 'anal intercourse',
    'bang': 'sexual intercourse',
    'screw': 'sexual intercourse',
    'fuck': 'sexual intercourse',
    'hump': 'sexual intercourse',
    'nookie': 'sexual intercourse',
    'claptrap': 'gonorrhea',
    'bum': 'buttocks',
    'booty': 'buttocks',
    'knock up': 'impregnate',
    'preggers': 'pregnant',
    'bun in the oven': 'pregnant',
    'yeast': 'yeast infection',
    'jock itch': 'tinea cruris',
    'smeg': 'smegma',
    'blue waffle': 'vaginal infection (colloquial, non-specific)',
    'morning wood': 'nocturnal penile tumescence',
    'pube': 'pubic hair',
    'bush': 'pubic hair',
    'fuzz': 'pubic hair'
}

def normalize_query(words: list) -> list:
    """Normalize slang or derogatory terms in the input words to formal medical terms."""
    try:
        normalized_words = []
        for word in words:
            # Case-insensitive normalization
            normalized = slang_to_formal.get(word.lower(), word)
            normalized_words.append(normalized)
            if normalized != word:
                logger.info(f"Normalized '{word}' to '{normalized}'")
        return normalized_words
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return words

def correct_spelling(query: str) -> str:
    """Correct spelling errors in the input query, then normalize slang terms."""
    try:
        # Split query into words
        words = query.split()
        corrected_words = []

        # Step 1: Correct spelling errors
        for word in words:
            if word.lower() not in spell and not word.isdigit():
                corrected = spell.correction(word)
                if corrected != word and corrected is not None:
                    logger.info(f"Corrected '{word}' to '{corrected}'")
                    corrected_words.append(corrected)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        # Step 2: Normalize slang terms in the corrected words
        normalized_words = normalize_query(corrected_words)
        corrected_query = ' '.join(normalized_words)

        # Log the overall transformation
        if corrected_query != query:
            logger.info(f"Processed query: '{query}' -> '{corrected_query}'")
        else:
            logger.info("No spelling corrections or normalizations needed")
        return corrected_query

    except Exception as e:
        logger.error(f"Spelling correction error: {e}")
        return query

# Example usage (for testing purposes)
if __name__ == "__main__":
    test_queries = [
        "I have a migrane and took paracetomol",
        "My dick hurts and I have the clap",
        "Feeling shits and pissing blood",
        "High blod presure and diabeetes"
    ]
    for query in test_queries:
        result = correct_spelling(query)
        print(f"Original: {query}\nCorrected: {result}\n")