from typing import Optional
from pydantic import BaseModel, Field
from .ner_layer import ExtractedEntities
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DrugDatabaseSearch(BaseModel):
    """Search parameters for Neo4j database of medications."""
    drug_name_search: Optional[str] = Field(None, description="Search query for medication brand name.")
    therapeutic_class_search: Optional[str] = Field(None, description="Search query for therapeutic classes.")
    composition_search: Optional[str] = Field(None, description="Search query for active ingredients/composition.")
    formulation_search: Optional[str] = Field(None, description="Search query for medication forms.")
    dosage_value_min: Optional[float] = Field(None, description="Minimum dosage value filter, inclusive.")
    dosage_value_max: Optional[float] = Field(None, description="Maximum dosage value filter, inclusive.")
    dosage_unit: Optional[str] = Field(None, description="Filter by specific dosage unit.")
    packaging_search: Optional[str] = Field(None, description="Search query for packaging types.")
    price_min: Optional[float] = Field(None, description="Minimum price filter in currency units, inclusive.")
    price_max: Optional[float] = Field(None, description="Maximum price filter in currency units, inclusive.")
    sort_by: Optional[str] = Field(None, description="Sorting criteria for results.")
    limit: Optional[int] = Field(10, description="Maximum number of results to return. Default is 10. Maximum is 15.")
    use_search: Optional[str] = Field(None, description="Search query for medical uses.")  # Add this field

    def to_cypher_query(self) -> str:
        """Convert search parameters to a Cypher query for Neo4j."""
        query = "MATCH (b:Brand)"
        therapeutic_var = "tc"
        composition_var = "c"
        dosage_var = "d"
        presentation_var = "p"
        uses_var = "u"
        considerations_var = "con"
        contraindications_var = "contra"

        has_therapeutic = False
        has_composition = False
        has_dosage = False
        has_presentation = False
        has_use = False

        if self.therapeutic_class_search:
            query += f"\nMATCH (b)-[:IN_CLASS]->({therapeutic_var}:TherapeuticClass)"
            has_therapeutic = True
        if self.composition_search:
            query += f"\nMATCH (b)-[:CONTAINS]->({composition_var}:Composition)"
            has_composition = True
        if self.dosage_value_min is not None or self.dosage_value_max is not None or self.dosage_unit:
            query += f"\nMATCH (b)-[:HAS_DOSAGE]->({dosage_var}:Dosage)"
            has_dosage = True
        if self.formulation_search or self.packaging_search:
            query += f"\nMATCH (b)-[:HAS_PRESENTATION]->({presentation_var}:Presentation)"
            has_presentation = True
        if self.use_search:
            query += f"\nMATCH (b)-[:USED_FOR]->({uses_var}:Use)"
            has_use = True

        if not has_therapeutic:
            query += f"\nOPTIONAL MATCH (b)-[:IN_CLASS]->({therapeutic_var}:TherapeuticClass)"
        if not has_composition:
            query += f"\nOPTIONAL MATCH (b)-[:CONTAINS]->({composition_var}:Composition)"
        if not has_dosage:
            query += f"\nOPTIONAL MATCH (b)-[:HAS_DOSAGE]->({dosage_var}:Dosage)"
        if not has_presentation:
            query += f"\nOPTIONAL MATCH (b)-[:HAS_PRESENTATION]->({presentation_var}:Presentation)"
        if not has_use:
            query += f"\nOPTIONAL MATCH (b)-[:USED_FOR]->({uses_var}:Use)"

        query += f"\nOPTIONAL MATCH (b)-[:HAS_CONSIDERATION]->({considerations_var}:Consideration)"
        query += f"\nOPTIONAL MATCH (b)-[:HAS_CONTRAINDICATION]->({contraindications_var}:Contraindication)"

        conditions = []
        if self.drug_name_search:
            conditions.append(f"b.name =~ '(?i).*{self.drug_name_search}.*'")
        if self.therapeutic_class_search:
            conditions.append(f"{therapeutic_var}.name =~ '(?i).*{self.therapeutic_class_search}.*'")
        if self.composition_search:
            conditions.append(f"{composition_var}.name =~ '(?i).*{self.composition_search}.*'")
        if self.formulation_search:
            conditions.append(f"{presentation_var}.form =~ '(?i).*{self.formulation_search}.*'")
        if self.dosage_value_min is not None:
            conditions.append(f"{dosage_var}.value >= {self.dosage_value_min}")
        if self.dosage_value_max is not None:
            conditions.append(f"{dosage_var}.value <= {self.dosage_value_max}")
        if self.dosage_unit:
            conditions.append(f"{dosage_var}.unit =~ '(?i).*{self.dosage_unit}.*'")
        if self.packaging_search:
            conditions.append(f"{presentation_var}.packaging =~ '(?i).*{self.packaging_search}.*'")
        if self.price_min is not None:
            conditions.append(f"b.price >= {self.price_min}")
        if self.price_max is not None:
            conditions.append(f"b.price <= {self.price_max}")
        if self.use_search:
            conditions.append(f"{uses_var}.name =~ '(?i).*{self.use_search}.*'")

        if conditions:
            query += "\nWHERE " + " AND ".join(conditions)

        query += f"\nRETURN b, {therapeutic_var}, {composition_var}, {dosage_var}, {presentation_var}, {uses_var}, {considerations_var}, {contraindications_var}"

        if self.sort_by == "price_asc":
            query += "\nORDER BY b.price ASC"
        elif self.sort_by == "price_desc":
            query += "\nORDER BY b.price DESC"
        elif self.sort_by == "name_asc":
            query += "\nORDER BY b.name ASC"
        elif self.sort_by == "name_desc":
            query += "\nORDER BY b.name DESC"

        query += f"\nLIMIT {self.limit}"
        return query

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
    """Layer for constructing database queries from extracted entities."""
    
    def entities_to_search_params(self, entities: ExtractedEntities) -> DrugDatabaseSearch:
        """Convert extracted entities to database search parameters."""
        search_params = {}
        if entities.brand:
            search_params["drug_name_search"] = entities.brand[0] if len(entities.brand) == 1 else "|".join(entities.brand)
        if entities.therapeutic_class:
            search_params["therapeutic_class_search"] = entities.therapeutic_class[0] if len(entities.therapeutic_class) == 1 else "|".join(entities.therapeutic_class)
        if entities.composition:
            search_params["composition_search"] = entities.composition[0] if len(entities.composition) == 1 else "|".join(entities.composition)
        if entities.form:
            search_params["formulation_search"] = entities.form[0] if len(entities.form) == 1 else "|".join(entities.form)
        if entities.dosage_value:
            if len(entities.dosage_value) == 1:
                search_params["dosage_value_min"] = entities.dosage_value[0]
                search_params["dosage_value_max"] = entities.dosage_value[0]
            elif len(entities.dosage_value) == 2:
                search_params["dosage_value_min"] = min(entities.dosage_value)
                search_params["dosage_value_max"] = max(entities.dosage_value)
        if entities.price_min and not entities.dosage_value:
            search_params["dosage_value_min"] = entities.price_min
        if entities.dosage_unit:
            search_params["dosage_unit"] = entities.dosage_unit[0]
        if entities.packaging:
            search_params["packaging_search"] = entities.packaging[0] if len(entities.packaging) == 1 else "|".join(entities.packaging)
        if entities.price_min:
            search_params["price_min"] = entities.price_min
        if entities.price_max:
            search_params["price_max"] = entities.price_max
        if entities.sort_preference:
            sort_mapping = {
                "cheapest": "price_asc",
                "most expensive": "price_desc",
                "alphabetical": "name_asc",
                "reverse alphabetical": "name_desc"
            }
            search_params["sort_by"] = sort_mapping.get(entities.sort_preference.lower())
        if entities.limit:
            search_params["limit"] = min(entities.limit, 15)
        if entities.use:
            search_params["use_search"] = entities.use[0] if len(entities.use) == 1 else "|".join(entities.use)
        return DrugDatabaseSearch(**search_params)