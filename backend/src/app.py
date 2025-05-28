from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from layers.hybrid_kg_layer import HybridPharmaceuticalKG
from layers.ner_layer import NamedEntityRecognizer
from layers.query_constructor_layer import QueryConstructorLayer
from layers.neo4j_layer import Neo4jLayer
from layers.question_answering_layer import QuestionAnsweringLayer
from layers.translation_layer import translate_query
from layers.spelling_correction_layer import correct_spelling
from layers.safety_check_layer import safety_check
from layers.prompt_template_conversion_layer import convert_to_template

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Debug environment variables
required_env_vars = [
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    "OPENROUTER_API_KEY", "OPENROUTER_API_URL",
    "NER_MODEL", "QA_MODEL"
]
for var in required_env_vars:
    value = os.getenv(var)
    logger.info(f"Env var {var}: {'Set' if value else 'Not set'}")
    if not value:
        logger.error(f"Missing required environment variable: {var}")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') # Use eventlet for better async handling

# Initialize LLM for NER
ner_llm = ChatOpenAI(
    model_name=os.getenv("NER_MODEL"),
    temperature=0,
    max_tokens=512,
    openai_api_base=os.getenv("OPENROUTER_API_URL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)

# Initialize layers
ner_layer = NamedEntityRecognizer(ner_llm)
query_constructor_layer = QueryConstructorLayer()
neo4j_layer = Neo4jLayer()
qa_layer = QuestionAnsweringLayer()

# Initialize HybridPharmaceuticalKG
try:
    hybrid_kg = HybridPharmaceuticalKG(
        csv_path=os.path.join('data', '3rd_database_translated.csv'),
        embedding_dim=64,
        walk_length=50,
        num_walks=5,
        batch_size=16,
        enable_text_embeddings=True
    )
    model_dir = os.path.join('hybrid_model_output')
    if os.path.exists(model_dir):
        logger.info("Loading pre-trained hybrid model...")
        try:
            hybrid_kg.load_model(model_dir)
        except Exception as e:
            logger.warning(f"Failed to load pre-trained model: {e}. Building new model...")
            hybrid_kg.preprocess_data()
            hybrid_kg.build_graph()
            hybrid_kg.generate_embeddings()
            hybrid_kg.build_faiss_indices()
            hybrid_kg.save_model(model_dir)
    else:
        logger.info("Building new hybrid model...")
        hybrid_kg.preprocess_data()
        hybrid_kg.build_graph()
        hybrid_kg.generate_embeddings()
        hybrid_kg.build_faiss_indices()
        hybrid_kg.save_model(model_dir)
except Exception as e:
    logger.error(f"Failed to initialize HybridPharmaceuticalKG: {e}")
    raise

@socketio.on('process_query')
def handle_process_query(data):
    """Process a chatbot query through the pipeline and emit step updates."""
    try:
        if not data or 'query' not in data:
            emit('error', {"error": "Missing query parameter"})
            return

        query = data['query']
        user_profile = data.get('userProfile', {})
        logger.info(f"Processing query: {query}")

        # Layer 1: Translation
        lang, translated_query = translate_query(query)
        emit('step_update', {'step': 'Translation', 'status': 'completed', 'result': translated_query})
        socketio.sleep(0)  # Force immediate flush
        
        # Layer 2: Spelling Correction
        corrected_query = correct_spelling(translated_query)
        emit('step_update', {'step': 'Spelling Correction', 'status': 'completed', 'result': corrected_query})
        socketio.sleep(0)  # Force immediate flush
        
        # Layer 3: NER Extraction
        entities = ner_layer.extract_entities(corrected_query)
        if not entities:
            emit('error', {"error": "Failed to extract entities"})
            return
        emit('step_update', {'step': 'NER Extraction', 'status': 'completed', 'result': entities.model_dump()})
        socketio.sleep(0)  # Force immediate flush

        # Layer 4: Query to Cypher
        search_params = query_constructor_layer.entities_to_search_params(entities)
        cypher_query = search_params.to_cypher_query()
        emit('step_update', {'step': 'Cypher Query', 'status': 'completed', 'result': cypher_query})
        socketio.sleep(0)  # Force immediate flush

        # Layer 5: Neo4j Extraction
        neo4j_results = neo4j_layer.execute_query(cypher_query)
        emit('step_update', {'step': 'Neo4j Extraction', 'status': 'completed', 'result': f"{len(neo4j_results)} results"})
        socketio.sleep(0)

        # Layer 6: Prompt Template Conversion & Knowledge Retrieval
        combined_prompt = convert_to_template(entities)
        hybrid_results = hybrid_kg.find_drugs_smart(query=combined_prompt, k=10)
        emit('step_update', {'step': 'Knowledge Retrieval', 'status': 'completed', 'result': f"{len(hybrid_results)} results"})
        socketio.sleep(0)

        # Layer 7: Safety Check
        filtered_neo4j, filtered_hybrid, flags = safety_check(neo4j_results, hybrid_results, user_profile, entities)
        emit('step_update', {'step': 'Safety Check', 'status': 'completed', 'result': f"{len(filtered_neo4j)} Neo4j, {len(filtered_hybrid)} hybrid results"})
        socketio.sleep(0)  # Force immediate flush

        # Layer 8: Question Answering
        answer = qa_layer.generate_answer(corrected_query, flags, filtered_neo4j, filtered_hybrid, lang)
        emit('step_update', {'step': 'Answer Generation', 'status': 'completed', 'result': 'Answer generated'})
        socketio.sleep(0)  # Force immediate flush

        # Final response
        response = {
            "content": answer,
            "entities": entities.model_dump(),
            "neo4j_results": filtered_neo4j,
            "hybrid_results": filtered_hybrid,
            "lang": lang,
            "metadata": {
                "processing_steps": [
                    {"step": "Translation", "result": translated_query},
                    {"step": "Spelling Correction", "result": corrected_query},
                    {"step": "NER Extraction", "result": entities.model_dump()},
                    {"step": "Cypher Query", "result": cypher_query},
                    {"step": "Neo4j Extraction", "result": f"{len(neo4j_results)} results"},
                    {"step": "Knowledge Retrieval", "result": f"{len(hybrid_results)} results"},
                    {"step": "Safety Check", "result": f"{len(filtered_neo4j)} Neo4j, {len(filtered_hybrid)} hybrid results"},
                    {"step": "Answer Generation", "result": "Answer generated"},
                ]
            }
        }
                
        emit('final_response', response)
        socketio.sleep(0)  # Force immediate flush

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        emit('error', {"error": "Internal server error", "details": str(e)})
        
        
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)