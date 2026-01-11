"""
ChEMBL MCP Client

Provides type-safe interface to ChEMBL MCP server with 27 specialized tools for
drug discovery, chemical informatics, and bioactivity analysis.

Based on ChEMBL database (https://www.ebi.ac.uk/chembl/) via REST API.
Server: https://github.com/augmented-nature/chembl-mcp-server
"""

from typing import Dict, Any, List, Optional
from .base import MCPSubprocessClient


class ChEMBLClient(MCPSubprocessClient):
    """
    ChEMBL MCP client with 27 tools for drug discovery and chemical informatics.

    Supports:
    - Compound search and structure retrieval
    - Target analysis and protein interaction
    - Bioactivity measurements and assay data
    - Drug development and clinical information
    - Chemical property analysis (ADMET, drug-likeness)
    - Advanced search and cross-referencing
    """

    def __init__(self, server_path: str):
        """
        Initialize ChEMBL MCP client.

        Args:
            server_path: Path to ChEMBL MCP server build/index.js
        """
        super().__init__(server_path, "ChEMBL", timeout=30)

    # =========================================================================
    # Core Chemical Search & Retrieval (5 tools)
    # =========================================================================

    async def search_compounds(
        self,
        query: str,
        limit: int = 25,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search the ChEMBL database for compounds by name, synonym, or identifier.

        Args:
            query: Search query (compound name, synonym, or ChEMBL ID)
            limit: Number of results to return (1-1000, default: 25)
            offset: Number of results to skip for pagination (default: 0)

        Returns:
            Dictionary containing:
                - molecules: List of compound objects with structure and properties
                - molecule_chembl_id: ChEMBL compound ID
                - pref_name: Preferred compound name
                - molecule_structures: SMILES, InChI, InChI Key
                - molecule_properties: MW, logP, HBA, HBD, PSA, Ro5 violations

        Example:
            >>> result = await client.search_compounds("aspirin", limit=5)
            >>> molecules = result['molecules']
            >>> print(f"Found {len(molecules)} compounds")
        """
        return await self.call_tool_with_retry("search_compounds", {
            "query": query,
            "limit": limit,
            "offset": offset
        })

    async def get_compound_info(self, chembl_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific compound by ChEMBL ID.

        Args:
            chembl_id: ChEMBL compound ID (e.g., CHEMBL25 for aspirin)

        Returns:
            Dictionary containing complete compound information:
                - molecule_chembl_id: ChEMBL ID
                - pref_name: Preferred name
                - molecule_type: Type (Small molecule, Protein, etc.)
                - molecule_structures: Canonical SMILES, InChI, InChI Key
                - molecule_properties: Molecular weight, logP, HBA, HBD, PSA, etc.
                - molecule_synonyms: Alternative names
                - atc_classifications: Anatomical Therapeutic Chemical codes
                - indication_class: Therapeutic indication

        Example:
            >>> info = await client.get_compound_info("CHEMBL25")
            >>> print(f"Name: {info['pref_name']}")
            >>> print(f"MW: {info['molecule_properties']['molecular_weight']}")
        """
        return await self.call_tool_with_retry("get_compound_info", {
            "chembl_id": chembl_id
        })

    async def search_by_inchi(self, inchi: str) -> Dict[str, Any]:
        """
        Find compounds by InChI key or InChI string.

        Args:
            inchi: InChI key or full InChI string

        Returns:
            Dictionary with matching molecules

        Example:
            >>> result = await client.search_by_inchi("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        """
        return await self.call_tool_with_retry("search_by_inchi", {
            "inchi": inchi
        })

    async def get_compound_structure(
        self,
        chembl_id: str,
        format: str = "smiles"
    ) -> Dict[str, Any]:
        """
        Retrieve chemical structure information in various formats.

        Args:
            chembl_id: ChEMBL compound ID
            format: Structure format (smiles, inchi, mol, sdf)

        Returns:
            Dictionary with structure in requested format

        Example:
            >>> structure = await client.get_compound_structure("CHEMBL25", "smiles")
        """
        return await self.call_tool_with_retry("get_compound_structure", {
            "chembl_id": chembl_id,
            "format": format
        })

    async def search_similar_compounds(
        self,
        chembl_id: str,
        similarity: float = 0.7,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Find chemically similar compounds using Tanimoto similarity.

        Args:
            chembl_id: Reference ChEMBL compound ID
            similarity: Tanimoto similarity threshold (0.0-1.0, default: 0.7)
            limit: Maximum number of results (default: 10)

        Returns:
            Dictionary with similar compounds and similarity scores

        Example:
            >>> similar = await client.search_similar_compounds("CHEMBL25", similarity=0.8)
        """
        return await self.call_tool_with_retry("search_similar_compounds", {
            "chembl_id": chembl_id,
            "similarity": similarity,
            "limit": limit
        })

    # =========================================================================
    # Target Analysis & Drug Discovery (5 tools)
    # =========================================================================

    async def search_targets(
        self,
        query: str,
        target_type: Optional[str] = None,
        organism: str = "Homo sapiens",
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Search for biological targets by name or type.

        Args:
            query: Target name or search query
            target_type: Target type filter (SINGLE PROTEIN, PROTEIN COMPLEX, etc.)
            organism: Organism filter (default: "Homo sapiens")
            limit: Number of results (1-1000, default: 25)

        Returns:
            Dictionary containing:
                - targets: List of target objects
                - target_chembl_id: ChEMBL target ID
                - pref_name: Preferred target name
                - target_type: Type of target
                - organism: Species
                - target_components: Protein components with UniProt IDs

        Example:
            >>> targets = await client.search_targets("kinase", limit=10)
            >>> targets = await client.search_targets("AXL", organism="Homo sapiens")
        """
        params = {
            "query": query,
            "organism": organism,
            "limit": limit
        }
        if target_type:
            params["target_type"] = target_type
        return await self.call_tool_with_retry("search_targets", params)

    async def get_target_info(self, target_chembl_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific target by ChEMBL target ID.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., CHEMBL2095173)

        Returns:
            Dictionary with complete target information including:
                - target_chembl_id: ChEMBL ID
                - pref_name: Preferred name
                - target_type: Type classification
                - organism: Species
                - target_components: Protein components with sequences and UniProt IDs
                - species_group_flag: Organism grouping

        Example:
            >>> info = await client.get_target_info("CHEMBL2095173")
        """
        return await self.call_tool_with_retry("get_target_info", {
            "target_chembl_id": target_chembl_id
        })

    async def get_target_compounds(
        self,
        target_chembl_id: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get compounds that have been tested against a specific target.

        Args:
            target_chembl_id: ChEMBL target ID
            limit: Maximum number of compounds (default: 50)

        Returns:
            Dictionary with:
                - compounds: List of compounds tested against target
                - bioactivity_data: Activity measurements (IC50, Ki, etc.)

        Example:
            >>> compounds = await client.get_target_compounds("CHEMBL2095173")
        """
        return await self.call_tool_with_retry("get_target_compounds", {
            "target_chembl_id": target_chembl_id,
            "limit": limit
        })

    async def search_by_uniprot(
        self,
        uniprot_id: str,
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Find ChEMBL targets by UniProt accession number.

        Critical for gene symbol → ChEMBL target mapping.

        Args:
            uniprot_id: UniProt accession (e.g., P30530 for AXL)
            limit: Number of results (1-1000, default: 25)

        Returns:
            Dictionary with matching ChEMBL targets

        Example:
            >>> targets = await client.search_by_uniprot("P30530")  # AXL
        """
        return await self.call_tool_with_retry("search_by_uniprot", {
            "uniprot_id": uniprot_id,
            "limit": limit
        })

    async def get_target_pathways(self, target_chembl_id: str) -> Dict[str, Any]:
        """
        Get biological pathways associated with a target.

        Args:
            target_chembl_id: ChEMBL target ID

        Returns:
            Dictionary with associated pathways

        Example:
            >>> pathways = await client.get_target_pathways("CHEMBL2095173")
        """
        return await self.call_tool_with_retry("get_target_pathways", {
            "target_chembl_id": target_chembl_id
        })

    # =========================================================================
    # Bioactivity & Assay Data (5 tools)
    # =========================================================================

    async def search_activities(
        self,
        target_chembl_id: Optional[str] = None,
        assay_chembl_id: Optional[str] = None,
        molecule_chembl_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Search bioactivity measurements and assay results.

        Critical for druggability assessment and drug discovery.

        Args:
            target_chembl_id: Filter by target
            assay_chembl_id: Filter by assay
            molecule_chembl_id: Filter by compound
            activity_type: Activity type (IC50, Ki, EC50, Kd, etc.)
            limit: Number of results (1-1000, default: 100)

        Returns:
            Dictionary containing:
                - activities: List of bioactivity measurements
                - activity_id: Unique activity ID
                - standard_type: Activity type (IC50, Ki, etc.)
                - standard_value: Numeric value
                - standard_units: Units (nM, uM, etc.)
                - standard_relation: Relation (=, <, >, ~)
                - assay_chembl_id: Associated assay
                - target_chembl_id: Associated target
                - molecule_chembl_id: Associated compound

        Example:
            >>> # Get all IC50 data for a target
            >>> activities = await client.search_activities(
            ...     target_chembl_id="CHEMBL2095173",
            ...     activity_type="IC50"
            ... )
        """
        params = {"limit": limit}
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id
        if assay_chembl_id:
            params["assay_chembl_id"] = assay_chembl_id
        if molecule_chembl_id:
            params["molecule_chembl_id"] = molecule_chembl_id
        if activity_type:
            params["activity_type"] = activity_type
        return await self.call_tool_with_retry("search_activities", params)

    async def get_assay_info(self, chembl_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific assay.

        Args:
            chembl_id: ChEMBL assay ID (e.g., CHEMBL1217643)

        Returns:
            Dictionary with assay details:
                - assay_chembl_id: ChEMBL ID
                - description: Assay description
                - assay_type: Type (B=Binding, F=Functional, etc.)
                - assay_organism: Test organism
                - confidence_score: Data quality score

        Example:
            >>> assay = await client.get_assay_info("CHEMBL1217643")
        """
        return await self.call_tool_with_retry("get_assay_info", {
            "chembl_id": chembl_id
        })

    async def search_by_activity_type(
        self,
        activity_type: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        units: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Find bioactivity data by specific activity type and value range.

        Args:
            activity_type: Activity type (IC50, Ki, EC50, Kd)
            min_value: Minimum activity value
            max_value: Maximum activity value
            units: Units filter (nM, uM, etc.)
            limit: Number of results (1-1000, default: 50)

        Returns:
            Dictionary with filtered bioactivity data

        Example:
            >>> # Find high-potency IC50 data (<100nM)
            >>> activities = await client.search_by_activity_type(
            ...     "IC50",
            ...     max_value=100,
            ...     units="nM"
            ... )
        """
        params = {
            "activity_type": activity_type,
            "limit": limit
        }
        if min_value is not None:
            params["min_value"] = min_value
        if max_value is not None:
            params["max_value"] = max_value
        if units:
            params["units"] = units
        return await self.call_tool_with_retry("search_by_activity_type", params)

    async def get_dose_response(
        self,
        molecule_chembl_id: str,
        target_chembl_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get dose-response data and activity profiles for compounds.

        Args:
            molecule_chembl_id: ChEMBL compound ID
            target_chembl_id: Optional target filter

        Returns:
            Dictionary with dose-response curves and IC50/EC50 data

        Example:
            >>> dr_data = await client.get_dose_response("CHEMBL25")
        """
        params = {"molecule_chembl_id": molecule_chembl_id}
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id
        return await self.call_tool_with_retry("get_dose_response", params)

    async def compare_activities(
        self,
        molecule_chembl_ids: List[str],
        target_chembl_id: Optional[str] = None,
        activity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare bioactivity data across multiple compounds or targets.

        Args:
            molecule_chembl_ids: Array of ChEMBL compound IDs (2-10)
            target_chembl_id: Optional target for comparison
            activity_type: Optional activity type for comparison

        Returns:
            Dictionary with comparative bioactivity analysis

        Example:
            >>> comparison = await client.compare_activities(
            ...     ["CHEMBL25", "CHEMBL59", "CHEMBL1642"],
            ...     activity_type="IC50"
            ... )
        """
        params = {"molecule_chembl_ids": molecule_chembl_ids}
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id
        if activity_type:
            params["activity_type"] = activity_type
        return await self.call_tool_with_retry("compare_activities", params)

    # =========================================================================
    # Drug Development & Clinical Data (4 tools)
    # =========================================================================

    async def search_drugs(
        self,
        query: str,
        development_phase: Optional[str] = None,
        therapeutic_area: Optional[str] = None,
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Search for approved drugs and clinical candidates.

        Args:
            query: Drug name or search query
            development_phase: Filter (e.g., "Approved", "Phase III")
            therapeutic_area: Therapeutic area filter
            limit: Number of results (1-1000, default: 25)

        Returns:
            Dictionary with drug information including clinical status

        Example:
            >>> drugs = await client.search_drugs("cancer", development_phase="Approved")
        """
        params = {
            "query": query,
            "limit": limit
        }
        if development_phase:
            params["development_phase"] = development_phase
        if therapeutic_area:
            params["therapeutic_area"] = therapeutic_area
        return await self.call_tool_with_retry("search_drugs", params)

    async def get_drug_info(self, drug_chembl_id: str) -> Dict[str, Any]:
        """
        Get drug development status and clinical trial information.

        Args:
            drug_chembl_id: ChEMBL drug ID

        Returns:
            Dictionary with drug development and clinical data

        Example:
            >>> drug_info = await client.get_drug_info("CHEMBL25")
        """
        return await self.call_tool_with_retry("get_drug_info", {
            "drug_chembl_id": drug_chembl_id
        })

    async def search_drug_indications(
        self,
        query: str,
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Search for therapeutic indications and disease areas.

        Args:
            query: Indication or disease area query
            limit: Number of results (default: 25)

        Returns:
            Dictionary with therapeutic indications

        Example:
            >>> indications = await client.search_drug_indications("breast cancer")
        """
        return await self.call_tool_with_retry("search_drug_indications", {
            "query": query,
            "limit": limit
        })

    async def get_mechanism_of_action(self, drug_chembl_id: str) -> Dict[str, Any]:
        """
        Get mechanism of action and target interaction data.

        Args:
            drug_chembl_id: ChEMBL drug ID

        Returns:
            Dictionary with mechanism of action details

        Example:
            >>> moa = await client.get_mechanism_of_action("CHEMBL25")
        """
        return await self.call_tool_with_retry("get_mechanism_of_action", {
            "drug_chembl_id": drug_chembl_id
        })

    # =========================================================================
    # Chemical Property Analysis (4 tools)
    # =========================================================================

    async def analyze_admet_properties(self, chembl_id: str) -> Dict[str, Any]:
        """
        Analyze ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity).

        Args:
            chembl_id: ChEMBL compound ID

        Returns:
            Dictionary with ADMET property predictions

        Example:
            >>> admet = await client.analyze_admet_properties("CHEMBL25")
        """
        return await self.call_tool_with_retry("analyze_admet_properties", {
            "chembl_id": chembl_id
        })

    async def calculate_descriptors(self, chembl_id: str) -> Dict[str, Any]:
        """
        Calculate molecular descriptors and physicochemical properties.

        Args:
            chembl_id: ChEMBL compound ID

        Returns:
            Dictionary with molecular descriptors (MW, logP, TPSA, etc.)

        Example:
            >>> descriptors = await client.calculate_descriptors("CHEMBL25")
        """
        return await self.call_tool_with_retry("calculate_descriptors", {
            "chembl_id": chembl_id
        })

    async def predict_solubility(self, chembl_id: str) -> Dict[str, Any]:
        """
        Predict aqueous solubility and permeability properties.

        Args:
            chembl_id: ChEMBL compound ID

        Returns:
            Dictionary with solubility predictions

        Example:
            >>> solubility = await client.predict_solubility("CHEMBL25")
        """
        return await self.call_tool_with_retry("predict_solubility", {
            "chembl_id": chembl_id
        })

    async def assess_drug_likeness(self, chembl_id: str) -> Dict[str, Any]:
        """
        Assess drug-likeness using Lipinski Rule of Five and other metrics.

        Critical for druggability assessment in Scenario 2.

        Args:
            chembl_id: ChEMBL compound ID

        Returns:
            Dictionary with drug-likeness assessment:
                - lipinski_compliant: Boolean
                - ro5_violations: Number of violations
                - molecular_weight: MW (≤500)
                - alogp: LogP (≤5)
                - hbd: H-bond donors (≤5)
                - hba: H-bond acceptors (≤10)

        Example:
            >>> drug_like = await client.assess_drug_likeness("CHEMBL25")
            >>> if drug_like['lipinski_compliant']:
            ...     print("Drug-like compound")
        """
        return await self.call_tool_with_retry("assess_drug_likeness", {
            "chembl_id": chembl_id
        })

    # =========================================================================
    # Advanced Search & Cross-References (4 tools)
    # =========================================================================

    async def substructure_search(
        self,
        smiles: str,
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Find compounds containing specific substructures.

        Args:
            smiles: SMILES string for substructure query
            limit: Number of results (default: 25)

        Returns:
            Dictionary with matching compounds

        Example:
            >>> matches = await client.substructure_search("c1ccccc1")  # Benzene ring
        """
        return await self.call_tool_with_retry("substructure_search", {
            "smiles": smiles,
            "limit": limit
        })

    async def batch_compound_lookup(self, chembl_ids: List[str]) -> Dict[str, Any]:
        """
        Process multiple ChEMBL IDs efficiently (batch operation).

        Critical for Scenario 6 drug repurposing with multiple compounds.

        Args:
            chembl_ids: Array of ChEMBL compound IDs (1-50)

        Returns:
            Dictionary with compound information for all IDs

        Example:
            >>> compounds = await client.batch_compound_lookup([
            ...     "CHEMBL25", "CHEMBL59", "CHEMBL1642"
            ... ])
        """
        return await self.call_tool_with_retry("batch_compound_lookup", {
            "chembl_ids": chembl_ids
        })

    async def get_external_references(self, chembl_id: str) -> Dict[str, Any]:
        """
        Get links to external databases (PubChem, DrugBank, PDB, etc.).

        Args:
            chembl_id: ChEMBL ID (compound, target, or assay)

        Returns:
            Dictionary with external database cross-references

        Example:
            >>> refs = await client.get_external_references("CHEMBL25")
        """
        return await self.call_tool_with_retry("get_external_references", {
            "chembl_id": chembl_id
        })

    async def advanced_search(
        self,
        filters: Dict[str, Any],
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Complex queries with multiple chemical and biological filters.

        Args:
            filters: Dictionary of filter criteria (MW range, logP, activity, etc.)
            limit: Number of results (default: 25)

        Returns:
            Dictionary with filtered search results

        Example:
            >>> results = await client.advanced_search({
            ...     "molecular_weight_min": 200,
            ...     "molecular_weight_max": 500,
            ...     "alogp_max": 5,
            ...     "ro5_violations": 0
            ... })
        """
        return await self.call_tool_with_retry("advanced_search", {
            "filters": filters,
            "limit": limit
        })
