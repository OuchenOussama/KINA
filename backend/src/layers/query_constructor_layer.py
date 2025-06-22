from typing import Optional, List
import json
import re
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from LLM response text."""
    try:
        response_text = response_text.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Response text: {response_text}")
        return None


class ExtractedEntities(BaseModel):
    """Named entities extracted from user queries about medications."""
    
    brand: Optional[List[str]] = Field(None, description="Brand names of medications (e.g., Tylenol, Advil, Aspirin)")
    composition: Optional[List[str]] = Field(None, description="Active ingredients/compositions (e.g., paracetamol, ibuprofen, amoxicillin)")
    consideration: Optional[List[str]] = Field(None, description="Medical considerations or warnings (e.g., 'take with food', 'avoid alcohol')")
    contraindication: Optional[List[str]] = Field(None, description="Contraindications or conditions where medication should be avoided")
    dosage_value: Optional[List[float]] = Field(None, description="Dosage values as numbers (e.g., 500, 1000, 2.5)")
    dosage_unit: Optional[List[str]] = Field(None, description="Dosage units (e.g., mg, ml, g, units)")
    packaging: Optional[List[str]] = Field(None, description="Packaging types (e.g., box, bottle, blister pack, vial)")
    form: Optional[List[str]] = Field(None, description="Medication forms (e.g., tablet, syrup, injection, capsule, cream)")
    therapeutic_class: Optional[List[str]] = Field(None, description="Therapeutic classes (e.g., antibiotics, analgesics, antihypertensives)")
    use: Optional[List[str]] = Field(None, description="Medical uses or indications (e.g., 'for pain relief', 'to treat infection')")
    price: Optional[List[float]] = Field(None, description="Specific price values mentioned (e.g., 'costs 25 dollars')")
    price_min: Optional[float] = Field(None, description="Minimum price mentioned (e.g., 'more than 10 dollars')")
    price_max: Optional[float] = Field(None, description="Maximum price mentioned (e.g., 'cheaper than 20 dollars')")
    sort_preference: Optional[str] = Field(None, description="Sorting preference (e.g., 'cheapest', 'most expensive', 'alphabetical')")
    limit: Optional[int] = Field(None, description="Number of results requested (e.g., 'show me 5 medications')")


class DrugDatabaseSearch(BaseModel):
    """Search parameters for Neo4j database of medications."""
    drug_name_search: Optional[str] = Field(None, description="Search query for medication brand name")
    therapeutic_class_search: Optional[str] = Field(None, description="Search query for therapeutic classes")
    composition_search: Optional[str] = Field(None, description="Search query for active ingredients")
    formulation_search: Optional[str] = Field(None, description="Search query for medication forms")
    dosage_value_min: Optional[float] = Field(None, description="Minimum dosage value filter")
    dosage_value_max: Optional[float] = Field(None, description="Maximum dosage value filter")
    dosage_unit: Optional[str] = Field(None, description="Filter by specific dosage unit")
    packaging_search: Optional[str] = Field(None, description="Search query for packaging types")
    price_min: Optional[float] = Field(None, description="Minimum price filter")
    price_max: Optional[float] = Field(None, description="Maximum price filter")
    sort_by: Optional[str] = Field(None, description="Sorting criteria for results")
    limit: Optional[int] = Field(10, description="Maximum number of results to return")
    use_search: Optional[str] = Field(None, description="Search query for medical uses")

    def to_cypher_query(self) -> str:
        """Convert search parameters to a Cypher query for Neo4j."""

        query_parts = ["MATCH (b:Brand)"]
        where_conditions = []

        relationships_added = set()

        if self.therapeutic_class_search:
            query_parts.append("MATCH (b)-[:BELONGS_TO_CLASS]->(tc:TherapeuticClass)")
            relationships_added.add("tc")
            where_conditions.append(f"toLower(tc.name) CONTAINS toLower('{self.therapeutic_class_search}')")

        if self.composition_search:
            query_parts.append("MATCH (b)-[:HAS_COMPOSITION]->(c:Composition)")
            relationships_added.add("c")
            where_conditions.append(f"toLower(c.name) CONTAINS toLower('{self.composition_search}')")

        if (self.dosage_value_min is not None or
            self.dosage_value_max is not None or
            self.dosage_unit):
            query_parts.append("MATCH (b)-[:HAS_DOSAGE]->(d:Dosage)")
            relationships_added.add("d")

            if self.dosage_value_min is not None:
                where_conditions.append(f"d.value >= {self.dosage_value_min}")
            if self.dosage_value_max is not None:
                where_conditions.append(f"d.value <= {self.dosage_value_max}")
            if self.dosage_unit:
                where_conditions.append(f"toLower(d.unit) CONTAINS toLower('{self.dosage_unit}')")

        if self.formulation_search or self.packaging_search:
            query_parts.append("MATCH (b)-[:HAS_PRESENTATION]->(p:Presentation)")
            relationships_added.add("p")

            if self.formulation_search:
                where_conditions.append(f"toLower(p.form) CONTAINS toLower('{self.formulation_search}')")
            if self.packaging_search:
                where_conditions.append(f"toLower(p.packaging) CONTAINS toLower('{self.packaging_search}')")

        if self.use_search:
            query_parts.append("MATCH (b)-[:TREATS]->(u:Use)")
            relationships_added.add("u")
            where_conditions.append(f"toLower(u.name) CONTAINS toLower('{self.use_search}')")

        if self.drug_name_search:
            where_conditions.append(f"toLower(b.name) CONTAINS toLower('{self.drug_name_search}')")

        if self.price_min is not None:
            where_conditions.append(f"b.price >= {self.price_min}")
        if self.price_max is not None:
            where_conditions.append(f"b.price <= {self.price_max}")

        optional_relationships = []

        if "tc" not in relationships_added:
            optional_relationships.append("OPTIONAL MATCH (b)-[:BELONGS_TO_CLASS]->(tc:TherapeuticClass)")
        if "c" not in relationships_added:
            optional_relationships.append("OPTIONAL MATCH (b)-[:HAS_COMPOSITION]->(c:Composition)")
        if "d" not in relationships_added:
            optional_relationships.append("OPTIONAL MATCH (b)-[:HAS_DOSAGE]->(d:Dosage)")
        if "p" not in relationships_added:
            optional_relationships.append("OPTIONAL MATCH (b)-[:HAS_PRESENTATION]->(p:Presentation)")
        if "u" not in relationships_added:
            optional_relationships.append("OPTIONAL MATCH (b)-[:TREATS]->(u:Use)")

        optional_relationships.extend([
            "OPTIONAL MATCH (b)-[:REQUIRES_CONSIDERATION]->(con:Consideration)",
            "OPTIONAL MATCH (b)-[:IS_CONTRAINDICATED_FOR]->(contra:Contraindication)"
        ])

        query_parts.extend(optional_relationships)

        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        return_clause = """
RETURN DISTINCT b.name as brand_name,
       b.price as price,
       collect(DISTINCT tc.name) as therapeutic_classes,
       collect(DISTINCT c.name) as compositions,
       collect(DISTINCT CASE WHEN d.value IS NOT NULL THEN toString(d.value) + ' ' + COALESCE(d.unit, '') ELSE null END) as dosages,
       collect(DISTINCT p.form) as forms,
       collect(DISTINCT p.packaging) as packaging,
       collect(DISTINCT u.name) as uses,
       collect(DISTINCT con.name) as considerations,
       collect(DISTINCT contra.name) as contraindications"""

        query_parts.append(return_clause)

        if self.sort_by == "price_asc":
            query_parts.append("ORDER BY b.price ASC")
        elif self.sort_by == "price_desc":
            query_parts.append("ORDER BY b.price DESC")
        elif self.sort_by == "name_asc":
            query_parts.append("ORDER BY b.name ASC")
        elif self.sort_by == "name_desc":
            query_parts.append("ORDER BY b.name DESC")
        else:
            query_parts.append("ORDER BY rand()")

        query_parts.append(f"LIMIT {min(self.limit, 15)}")

        final_query = "\n".join(query_parts)

        logger.info(f"Generated Cypher Query:\n{final_query}")

        return final_query

    def pretty_print(self) -> None:
        """Print all non-default field values in a readable format."""
        logger.info("Drug Database Search Parameters:")
        logger.info("---------------------------------")
        for field in self.__fields__:
            value = getattr(self, field)
            if value is not None and value != getattr(self.__fields__[field], "default", None):
                logger.info(f"{field}: {value}")
        logger.info("---------------------------------")
        logger.info("Generated Cypher Query:")
        logger.info(self.to_cypher_query())


class QueryConstructorLayer:
    """Enhanced layer for constructing database queries from extracted entities."""
    
    def entities_to_search_params(self, entities: ExtractedEntities) -> DrugDatabaseSearch:
        """Convert extracted entities to database search parameters."""
        search_params = {}

        if entities.brand and len(entities.brand) > 0:
            search_params["drug_name_search"] = entities.brand[0]

        if entities.therapeutic_class and len(entities.therapeutic_class) > 0:
            search_params["therapeutic_class_search"] = entities.therapeutic_class[0]

        if entities.composition and len(entities.composition) > 0:
            search_params["composition_search"] = entities.composition[0]

        if entities.form and len(entities.form) > 0:
            search_params["formulation_search"] = entities.form[0]

        if entities.dosage_value and len(entities.dosage_value) > 0:
            if len(entities.dosage_value) == 1:
                search_params["dosage_value_min"] = entities.dosage_value[0]
                search_params["dosage_value_max"] = entities.dosage_value[0]
            else:
                search_params["dosage_value_min"] = min(entities.dosage_value)
                search_params["dosage_value_max"] = max(entities.dosage_value)

        if entities.dosage_unit and len(entities.dosage_unit) > 0:
            search_params["dosage_unit"] = entities.dosage_unit[0]

        if entities.packaging and len(entities.packaging) > 0:
            search_params["packaging_search"] = entities.packaging[0]

        if entities.price_min is not None:
            search_params["price_min"] = entities.price_min

        if entities.price_max is not None:
            search_params["price_max"] = entities.price_max

        if entities.price and len(entities.price) > 0:
            if len(entities.price) == 1:
                search_params["price_min"] = entities.price[0]
                search_params["price_max"] = entities.price[0]

        if entities.sort_preference:
            sort_mapping = {
                "cheapest": "price_asc", 
                "price": "price_asc", 
                "most expensive": "price_desc",
                "expensive": "price_desc",
                "alphabetical": "name_asc",
                "alphabetically": "name_asc",
                "reverse alphabetical": "name_desc"
            }
            search_params["sort_by"] = sort_mapping.get(entities.sort_preference.lower(), "name_asc")

        if entities.limit and entities.limit > 0:
            search_params["limit"] = min(entities.limit, 15)

        if entities.use and len(entities.use) > 0:
            search_params["use_search"] = entities.use[0]

        return DrugDatabaseSearch(**search_params)