from neo4j import GraphDatabase
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jLayer:
    """Layer for interacting with Neo4j database."""
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j Aura."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Connected to Neo4j Aura")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def execute_query(self, cypher_query: str) -> list:
        """Execute a Cypher query and return results."""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                logger.info(f"Neo4j query returned {len(records)} records")
                for i, record in enumerate(records):
                    logger.debug(f"Record {i}: type={type(record)}, content={record}")
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return []

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")