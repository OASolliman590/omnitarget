<!-- 51552a12-a6a7-4b5e-aa25-a87eedbe4ce4 3e97eaef-7451-457f-bf7b-3b1752f6ebed -->
# OmniTarget Pipeline - Technical Implementation Plan

## Architecture Decisions

Based on documentation analysis and implementation choices:

- **MCP Communication**: Subprocess execution with JSON-RPC over stdin/stdout (1a)
- **Test Data**: Small sample datasets initially, document full benchmarks separately (2c)
- **MRA Implementation**: Start with simplified perturbation, migrate to full MRA (3b)
- **Validation Scope**: Layered validation - unit tests (Phase 1), integration (Phase 2), benchmarks (Phase 3) (4c)

## Phase 1: Core Infrastructure (Weeks 1-4)

### Week 1-2: MCP Client Layer

#### File: `src/mcp_clients/base.py`

Create base MCP client with subprocess communication:

```python
class MCPSubprocessClient:
    """Base class for all MCP server communication via subprocess"""
    - async start(): Launch Node.js MCP server process
    - async call_tool(tool_name, params): JSON-RPC communication
    - async stop(): Graceful shutdown
    - Error handling with retry logic (3 attempts, exponential backoff)
```

#### Files: `src/mcp_clients/{kegg,reactome,string,hpa}_client.py`

Implement specific clients inheriting from base:

- **KEGGClient**: 30 tools from test report (search_diseases, get_pathway_genes, etc.)
- **ReactomeClient**: 8 tools (search_pathways, find_pathways_by_gene, etc.)
- **STRINGClient**: 6 tools (get_interaction_network, get_functional_enrichment, etc.)
- **HPAClient**: 16 tools (search_proteins, get_tissue_expression, etc.)

Each client implements:

```python
async def tool_name(self, **kwargs) -> Dict:
    """Type-safe wrapper for MCP tool call"""
    return await self.call_tool('tool_name', kwargs)
```

#### File: `src/core/mcp_client_manager.py`

Unified manager for all MCP clients:

```python
class MCPClientManager:
    def __init__(self, mcp_base_path: str):
        # Initialize all 4 clients with paths from config
    async def start_all(): # Parallel startup
    async def stop_all(): # Graceful shutdown
    @contextmanager async def session(): # Context manager for lifecycle
```

**Config File**: `config/mcp_servers.json`

```json
{
  "kegg": {"path": "/Users/omara.soliman/Documents/mcp/kegg/build/index.js"},
  "reactome": {"path": "/Users/omara.soliman/Documents/mcp/reactome/build/index.js"},
  "string": {"path": "/Users/omara.soliman/Documents/mcp/string/build/index.js"},
  "hpa": {"path": "/Users/omara.soliman/Documents/mcp/proteinatlas/build/index.js"}
}
```

#### Testing Week 1-2:

**File**: `tests/unit/test_mcp_clients.py`

- Test each client's connection lifecycle (start/stop)
- Test 3-5 key tools per client with mock responses
- Test error handling (invalid params, server crash, timeout)
- Target: 40 unit tests, <5s runtime

**File**: `tests/fixtures/sample_mcp_responses.json`

Mock responses for deterministic testing

### Week 3-4: Data Standardization Layer

#### File: `src/models/data_models.py`

Pydantic models for all MCP outputs:

```python
class Disease(BaseModel):
    id: str
    name: str
    source_db: Literal['kegg', 'reactome']
    pathways: List[str]
    confidence: float

class Pathway(BaseModel):
    id: str
    name: str
    source_db: Literal['kegg', 'reactome']
    genes: List[str]
    hierarchy_level: Optional[int]

class Protein(BaseModel):
    gene_symbol: str
    uniprot_id: Optional[str]
    string_id: Optional[str]
    kegg_id: Optional[str]
    ensembl_id: Optional[str]

class Interaction(BaseModel):
    protein_a: str
    protein_b: str
    combined_score: float
    evidence_types: Dict[str, float]
    pathway_context: Optional[str]
```

Reference: Mature_development_plan.md Phase 2 for complete model specifications

#### File: `src/core/data_standardizer.py`

Normalize MCP outputs to unified models:

```python
class DataStandardizer:
    def standardize_kegg_disease(raw: Dict) -> Disease
    def standardize_reactome_pathway(raw: Dict) -> Pathway
    def standardize_string_interaction(raw: Dict) -> Interaction
    def merge_cross_database_results(results: Dict) -> UnifiedResult
```

#### File: `src/utils/id_mapping.py`

Cross-database identifier conversion:

```python
class IDMapper:
    def __init__(self):
        # Load mapping tables from KEGG convert_identifiers
    async def map_gene_symbol_to_ids(symbol: str) -> Protein
    async def map_uniprot_to_string(uniprot: str) -> str
    def validate_identifier(id: str, db: str) -> bool
```

Uses KEGG `convert_identifiers` tool and STRING `search_proteins` for validation

#### File: `src/core/validation.py`

Data quality checks from success_metrics.md:

```python
class DataValidator:
    def validate_disease_confidence(disease: Disease) -> bool:
        # success_metrics.md: Score ≥0.6
    def validate_interaction_confidence(interaction: Interaction) -> bool:
        # success_metrics.md: Median score ≥400
    def validate_expression_coverage(genes: List[str], expression_data: Dict) -> float:
        # success_metrics.md: ≥85% coverage
```

#### Testing Week 3-4:

**File**: `tests/unit/test_data_standardization.py`

- Test Pydantic model validation (invalid data rejection)
- Test standardization for each MCP output type
- Test ID mapping with known conversions (TP53 → P04637 → 9606.ENSP00000269305)
- Target: 50 unit tests, <3s runtime

**File**: `tests/fixtures/sample_disease_genes.json`

Small sample dataset: 15 breast cancer genes (TP53, BRCA1, BRCA2, ESR1, ERBB2, PIK3CA, AKT1, PTEN, CDH1, RB1, EGFR, MYC, CCND1, GATA3, FOXA1)

## Phase 2: Simulation Engine (Weeks 5-8)

### Week 5-6: Simplified Perturbation Simulator

#### File: `src/core/simulation/simple_simulator.py`

Confidence-weighted BFS propagation:

```python
class SimplePerturbationSimulator:
    def __init__(self, network: nx.Graph, mcp_data: Dict):
        self.network = network
        self.pathway_context = self._build_pathway_context(mcp_data)
        self.interaction_confidence = self._extract_string_confidence(mcp_data)
    
    async def simulate_perturbation(
        self,
        target_node: str,
        perturbation_strength: float = 0.9,
        max_depth: int = 3,
        confidence_threshold: float = 0.4
    ) -> SimulationResult:
        """
        Depth-limited propagation with confidence decay
        - Depth 1: strength × confidence × pathway_modifier
        - Depth 2: strength × confidence × pathway_modifier × 0.7
        - Depth 3: strength × confidence × pathway_modifier × 0.5
        """
    
    def _get_pathway_context(self, node_a: str, node_b: str) -> float:
        """
        From Mature_development_plan.md:
        - Same pathway: 1.2
        - Connected pathways: 0.8
        - Different pathways: 0.4
        """
    
    def _classify_effects(self, effects: Dict, target: str) -> Dict:
        """
        Classify nodes as:
        - direct_targets: immediate neighbors
        - downstream: reachable in ≤3 hops
        - feedback_loops: paths back to target
        """
```

#### File: `src/models/simulation_models.py`

```python
class SimulationConfig(BaseModel):
    mode: Literal['inhibit', 'activate']
    depth: int = 3
    propagation_factor: float = 0.7
    confidence_threshold: float = 0.4

class SimulationResult(BaseModel):
    target_node: str
    mode: str
    affected_nodes: Dict[str, float]  # node -> effect_strength
    direct_targets: List[str]
    downstream: List[str]
    network_impact: Dict[str, Any]
```

#### Testing Week 5-6:

**File**: `tests/unit/test_simple_simulation.py`

- Test single-target perturbation on small network (20 nodes)
- Test depth limiting (verify no effects beyond max_depth)
- Test confidence thresholding (low-confidence edges ignored)
- Test pathway context weighting
- Target: 30 unit tests, <2s runtime

**File**: `tests/fixtures/sample_string_network.json`

TP53 neighborhood: 20 proteins with STRING confidence scores

### Week 7-8: Full MRA Module (Advanced)

#### File: `src/core/simulation/mra_simulator.py`

Full Modular Response Analysis from Scienticaly_proven.md:

```python
class MRASimulator:
    def _build_response_matrix(self, tissue_context: Optional[str]) -> np.ndarray:
        """
        Build local response coefficient matrix R
        R[i,j] = influence of node j on node i
        
        Incorporates:
        - STRING confidence weighting (Level 1)
        - Pathway context modifier (Level 2)
        - Expression filtering (Level 3)
        """
    
    async def simulate_perturbation(
        self,
        target_node: str,
        perturbation_type: Literal['inhibit', 'activate'] = 'inhibit',
        perturbation_strength: float = 0.9,
        tissue_context: Optional[str] = None
    ) -> MRASimulationResult:
        """
        Steady-state MRA: (I - R)^-1 × perturbation_vector
        
        Handles:
        - Matrix inversion with regularization for singular matrices
        - Convergence validation
        - Feedback loop detection
        """
    
    def _classify_upstream_downstream(
        self, 
        steady_state: np.ndarray, 
        target_idx: int
    ) -> Dict:
        """
        Classify effects using:
        - Topological analysis (shortest paths)
        - Reactome pathway directionality
        - Response magnitude and sign
        """
```

#### File: `src/core/simulation/feedback_analyzer.py`

Detect and classify feedback loops:

```python
class FeedbackAnalyzer:
    def detect_feedback_loops(network: nx.DiGraph, target: str) -> List[FeedbackLoop]
    def classify_loop_type(loop: List[str]) -> Literal['positive', 'negative']
    def predict_compensatory_response(loop: FeedbackLoop) -> Dict
```

#### Testing Week 7-8:

**File**: `tests/integration/test_mra_simulation.py`

- Test MRA vs. simple simulator on same network (compare outputs)
- Test matrix convergence with known stable/unstable networks
- Test feedback loop detection (create test network with known loops)
- Validate against simplified version (should correlate ρ ≥0.6)
- Target: 20 integration tests, <30s runtime

## Phase 3: Scenario Implementation (Weeks 9-12)

### Week 9-10: Core Scenarios (1-3)

#### File: `src/scenarios/scenario_1_disease_network.py`

Based on Mature_development_plan.md Phase 1-5:

```python
class DiseaseNetworkScenario:
    async def execute(
        self,
        disease_query: str,
        tissue_context: Optional[str] = None
    ) -> DiseaseNetworkResult:
        """
        5-phase workflow from Mature_development_plan.md:
        1. Multi-database disease discovery (KEGG + Reactome)
        2. Pathway extraction with hierarchy (cross-validation)
        3. Context-aware network construction (STRING)
        4. Expression overlay (HPA)
        5. Functional enrichment analysis
        """
    
    async def _phase1_disease_discovery(self, query: str) -> Dict:
        # Parallel MCP calls: KEGG + Reactome
        # Cross-database concordance scoring (success_metrics.md: ≥70%)
    
    async def _phase3_network_construction(self, genes: List[str]) -> nx.Graph:
        # STRING network with confidence ≥400
        # Pathway context weighting (same: 1.2, connected: 0.8, different: 0.4)
        # Disease module identification (Louvain clustering)
```

#### File: `src/scenarios/scenario_2_target_analysis.py`

5-phase target-centric workflow:

```python
class TargetAnalysisScenario:
    async def execute(self, target_query: str) -> TargetAnalysisResult:
        """
        1. Multi-database target resolution (STRING + HPA + KEGG)
        2. Pathway membership analysis (Reactome primary)
        3. Interaction network (STRING + Reactome mechanistic)
        4. Expression profiling (HPA tissue-specific)
        5. Druggability assessment (KEGG drugs + localization)
        """
```

#### File: `src/scenarios/scenario_3_cancer_analysis.py`

Cancer-specific 5-phase workflow:

```python
class CancerAnalysisScenario:
    async def execute(
        self,
        cancer_type: str,
        tissue_context: str
    ) -> CancerAnalysisResult:
        """
        1. Cancer marker discovery (HPA prognostic markers)
        2. Cancer pathway discovery (KEGG + Reactome)
        3. Cancer network construction (STRING with marker weighting)
        4. Expression dysregulation (HPA cancer vs. normal)
        5. Target prioritization (multi-criteria scoring)
        """
    
    def _prioritize_cancer_targets(
        self,
        network: nx.Graph,
        markers: List[str],
        expression: Dict
    ) -> List[PrioritizedTarget]:
        """
        From Mature_development_plan.md Phase 5:
        Criteria weights:
        - Druggability: 0.25
        - Cancer specificity: 0.30
        - Network centrality: 0.20
        - Prognostic value: 0.15
        - Pathway impact: 0.10
        """
```

#### Testing Week 9-10:

**File**: `tests/integration/test_scenarios_1_3.py`

For each scenario:

- Test complete workflow with sample data (15 genes, 5 pathways)
- Validate success metrics from success_metrics.md:
    - Scenario 1: Cross-database concordance ≥70%, pathway coverage ≥80%
    - Scenario 2: ID mapping ≥95%, pathway precision ≥85%
    - Scenario 3: Marker validation ≥70%, driver overlap ≥50%
- Test error handling (invalid disease, unknown target, missing data)
- Target: 30 integration tests, <3min runtime

### Week 11-12: Advanced Scenarios (4-6)

#### File: `src/scenarios/scenario_4_mra_simulation.py`

Multi-target simulation with cross-validation:

```python
class MultiTargetSimulationScenario:
    async def execute(
        self,
        targets: List[str],
        disease_context: str,
        simulation_mode: Literal['simple', 'mra'] = 'simple'
    ) -> MultiTargetSimulationResult:
        """
        8-step workflow from OmniTarget_Development_Plan.md:
        1. Target resolution (STRING + HPA)
        2. Network context (Reactome pathways)
        3. Interaction validation (STRING)
        4. Expression validation (HPA)
        5. Pathway impact (Reactome participants)
        6. Network construction
        7. Simulation (simple or MRA)
        8. Impact assessment (STRING enrichment)
        """
    
    async def _run_simulation(
        self,
        network: nx.Graph,
        targets: List[str],
        mode: str
    ) -> SimulationResult:
        if mode == 'simple':
            simulator = SimplePerturbationSimulator(network, self.mcp_data)
        else:
            simulator = MRASimulator(network, self.mcp_data)
        
        return await simulator.simulate_perturbation(targets[0])
```

#### File: `src/scenarios/scenario_5_pathway_comparison.py`

Cross-database pathway validation:

```python
class PathwayComparisonScenario:
    async def execute(self, pathway_query: str) -> PathwayComparisonResult:
        """
        7-step workflow:
        1. Parallel search (KEGG + Reactome)
        2. ID mapping (KEGG convert_identifiers)
        3. Gene extraction
        4. Overlap analysis (Jaccard similarity, success_metrics.md: ≥0.4)
        5. Mechanistic details (Reactome reactions)
        6. Interaction validation (STRING)
        7. Expression context (HPA)
        """
```

#### File: `src/scenarios/scenario_6_drug_repurposing.py`

Drug repurposing with network validation:

```python
class DrugRepurposingScenario:
    async def execute(
        self,
        disease_query: str,
        tissue_context: str
    ) -> DrugRepurposingResult:
        """
        9-step workflow:
        1. Disease pathways (KEGG + Reactome)
        2. Pathway proteins extraction
        3. Known drugs (KEGG drug-target mappings)
        4. Target networks (STRING)
        5. Off-target analysis (network overlap)
        6. Expression validation (HPA)
        7. Cancer specificity (if applicable)
        8. MRA simulation of drug effects
        9. Pathway enrichment
        """
    
    def _calculate_repurposing_score(
        self,
        drug: str,
        simulation: SimulationResult,
        expression: Dict
    ) -> float:
        """
        Composite score:
        - Network impact (0-1)
        - Expression specificity (0-1)
        - Safety profile (0-1)
        """
```

#### Testing Week 11-12:

**File**: `tests/integration/test_scenarios_4_6.py`

- Scenario 4: Test single and multi-target simulation, validate MRA convergence
- Scenario 5: Test pathway overlap calculations, validate Jaccard ≥0.4
- Scenario 6: Test drug repurposing scoring, validate known repurposing cases
- Target: 25 integration tests, <5min runtime

#### File: `src/core/pipeline_orchestrator.py`

Main orchestrator for all scenarios:

```python
class OmniTargetPipeline:
    def __init__(self, config_path: str):
        self.mcp_manager = MCPClientManager(config_path)
        self.scenarios = {
            1: DiseaseNetworkScenario(self.mcp_manager),
            2: TargetAnalysisScenario(self.mcp_manager),
            3: CancerAnalysisScenario(self.mcp_manager),
            4: MultiTargetSimulationScenario(self.mcp_manager),
            5: PathwayComparisonScenario(self.mcp_manager),
            6: DrugRepurposingScenario(self.mcp_manager)
        }
    
    async def run_scenario(self, scenario_id: int, **kwargs):
        async with self.mcp_manager.session():
            return await self.scenarios[scenario_id].execute(**kwargs)
```

## Phase 4: Validation & Optimization (Weeks 13-16)

### Week 13-14: Performance Optimization

#### File: `src/utils/caching.py`

Intelligent caching for MCP calls:

```python
class MCPCache:
    def __init__(self, cache_dir: str, ttl: int = 3600):
        # File-based cache with TTL expiration
    
    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable,
        ttl_override: Optional[int] = None
    ) -> Any:
        """
        Cache MCP responses to avoid redundant calls
        - Disease/pathway lookups: 24hr TTL
        - Network data: 1hr TTL
        - Expression data: 12hr TTL
        """
```

#### Parallel MCP Calls Optimization:

Update all scenario files to use `asyncio.gather()` for parallel MCP queries:

```python
# Example from scenario_1_disease_network.py
kegg_diseases, reactome_pathways, hpa_markers = await asyncio.gather(
    self.kegg_client.search_diseases(query),
    self.reactome_client.find_pathways_by_disease(query),
    self.hpa_client.search_cancer_markers(query) if is_cancer else None
)
```

#### Memory Optimization:

```python
# src/utils/memory_manager.py
class MemoryManager:
    def stream_large_network(network: nx.Graph, chunk_size: int = 1000):
        """Process large networks in chunks to avoid memory overflow"""
    
    def lazy_load_expression(genes: List[str], batch_size: int = 100):
        """Load HPA expression data in batches"""
```

Performance Targets from OmniTarget_Development_Plan.md:

- Response time: <30s for single target simulation
- Memory usage: <2GB for 5000-node networks
- MCP tool success rate: >90%

### Week 15-16: Benchmark Validation

#### File: `tests/benchmarks/benchmark_plan.md`

Document full validation strategy:

```markdown
# Full Validation Benchmarks

## DREAM Challenge Data
- Source: synapse.org/DREAM
- Datasets: Network Inference Challenge 4, Drug Synergy Challenge
- Size: ~5GB
- Expected Metrics: Spearman ρ ≥0.5
- Implementation: Phase 4, Week 15

## TCGA Cancer Data
- Source: GDC Data Portal
- Cancer Types: BRCA, LUAD, COAD
- Sample Size: 100 patients per type
- Expected Metrics: 75% expression concordance
- Implementation: Phase 4, Week 15

## COSMIC Cancer Gene Census
- Source: cancer.sanger.ac.uk/cosmic
- Validation: Driver gene identification
- Expected Metrics: ≥50% overlap with identified drivers
- Implementation: Phase 4, Week 16

## Implementation Scripts
- download_dream.py: Automated DREAM data acquisition
- download_tcga.py: TCGA subset download via GDC API
- download_cosmic.py: COSMIC driver gene list
```

#### File: `tests/benchmarks/validate_full_pipeline.py`

Implement all validation metrics from success_metrics.md:

```python
@pytest.mark.benchmark
class TestBenchmarkValidation:
    def test_scenario1_disease_network_validation(self):
        """
        Validation criteria from success_metrics.md:
        - Cross-database concordance: ≥70%
        - Pathway coverage: ≥80%
        - Interaction confidence: median ≥400
        - Module enrichment: ≥80% modules with FDR <0.05
        """
    
    def test_scenario2_target_analysis_validation(self):
        """
        - ID mapping accuracy: ≥95%
        - Pathway precision/recall: ≥85%/≥80%
        - Expression reproducibility: Spearman ρ ≥0.7 vs GTEx
        - Druggability ROC-AUC: ≥0.75
        """
    
    def test_scenario3_cancer_analysis_validation(self):
        """
        - Prognostic marker validation: ≥70% in literature
        - Cancer hallmark enrichment: ≥4 of 10 hallmarks
        - Driver gene overlap: ≥50% with COSMIC
        - Differential expression concordance: ρ ≥0.6 vs TCGA
        """
    
    def test_scenario4_mra_simulation_validation(self):
        """
        - MRA convergence: max |Δx| <1e-6 within 1000 iterations
        - Known perturbation recapitulation: Spearman ρ ≥0.5 vs DREAM
        - Upstream/downstream accuracy: ≥75% agreement with Reactome
        - Confidence-effect correlation: monotonic increase Q1→Q4
        """
```

Target: 50 benchmark tests covering all success_metrics.md criteria, <4hrs runtime

#### File: `.github/workflows/test.yml`

CI/CD pipeline with layered testing:

```yaml
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=html
    # Runs on every commit (~10s)
  
  integration-tests:
    needs: unit-tests
    steps:
      - name: Install MCP servers
        run: |
          cd /Users/omara.soliman/Documents/mcp/kegg && npm install
          cd /Users/omara.soliman/Documents/mcp/reactome && npm install
          cd /Users/omara.soliman/Documents/mcp/string && npm install
          cd /Users/omara.soliman/Documents/mcp/proteinatlas && npm install
      - name: Run integration tests
        run: pytest tests/integration/ -v
    # Runs on every commit (~5min)
  
  benchmark-validation:
    if: github.event_name == 'schedule' || github.ref == 'refs/heads/main'
    steps:
      - name: Download benchmarks
        run: python tests/benchmarks/download_datasets.py
      - name: Run full validation
        run: pytest tests/benchmarks/ -v -m benchmark
    # Runs nightly or before release (~4hrs)
```

## Documentation

### File: `docs/api_reference.md`

Complete API documentation with examples:

- All MCP client methods with parameters and return types
- All scenario classes with usage examples
- Data models with field descriptions

### File: `docs/user_guide.md`

Usage-oriented guide:

- Installation instructions
- Quick start examples for each scenario
- Common workflows (disease analysis, target discovery, drug repurposing)
- Troubleshooting guide

### File: `docs/scientific_background.md`

Technical and scientific documentation:

- MRA mathematical foundation
- Validation methodology
- Success metrics interpretation
- Literature references

### File: `examples/basic_usage.py`

```python
from src.core.pipeline_orchestrator import OmniTargetPipeline

async def main():
    pipeline = OmniTargetPipeline('config/mcp_servers.json')
    
    # Scenario 1: Disease network
    result = await pipeline.run_scenario(
        1,
        disease_query="breast cancer",
        tissue_context="breast"
    )
    print(f"Found {len(result.network.nodes())} proteins in disease network")
```

### File: `examples/disease_analysis.py`

Complete disease analysis workflow

### File: `examples/drug_discovery.py`

Drug repurposing workflow example

## Configuration Files

### File: `requirements.txt`

```
pydantic>=2.0.0
networkx>=3.0
numpy>=1.24.0
scipy>=1.10.0
aiofiles>=23.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
```

### File: `setup.py`

Standard Python package setup

### File: `config/simulation_params.json`

```json
{
  "simple_perturbation": {
    "max_depth": 3,
    "confidence_threshold": 0.4,
    "decay_factor": 0.7,
    "pathway_weights": {
      "same_pathway": 1.2,
      "connected": 0.8,
      "different": 0.4
    }
  },
  "mra_simulation": {
    "max_iterations": 1000,
    "convergence_threshold": 1e-6,
    "regularization_factor": 0.01
  }
}
```

## Identified Gaps

### Gap 1: Reactome Pathway Directionality

**Location**: Mature_development_plan.md mentions using Reactome pathway directionality for upstream/downstream classification

**Need**: Clarify how to extract directionality from Reactome reactions

**Source**: `get_pathway_reactions` tool output format

### Gap 2: Expression Level Quantification

**Location**: Multiple scenarios use HPA expression levels

**Need**: Define exact mapping from HPA categorical levels (Not detected/Low/Medium/High) to numerical scores for calculations

**Proposed**: Not detected=0.0, Low=0.3, Medium=0.6, High=1.0

### Gap 3: Cancer Type to Tissue Mapping

**Location**: Scenario 3 requires mapping cancer types to normal tissues

**Need**: Create mapping table (e.g., "breast cancer" → "breast", "lung adenocarcinoma" → "lung")

**Implementation**: `src/utils/cancer_tissue_mapping.json`

### Gap 4: Benchmark Dataset Access

**Location**: success_metrics.md references DREAM, TCGA, COSMIC

**Need**: Confirm access credentials and API keys for automated downloads

**Action**: Document access requirements in tests/benchmarks/benchmark_plan.md

All gaps are non-blocking for Phase 1-2 implementation and can be addressed during Phase 3-4.

### To-dos

- [ ] Implement MCP client layer with subprocess communication (base + 4 specific clients)
- [ ] Create Pydantic data models and standardization layer
- [ ] Implement unit tests for MCP clients and data standardization
- [ ] Implement simplified perturbation simulator with confidence weighting
- [ ] Implement full MRA simulation engine with matrix operations
- [ ] Implement Scenarios 1-3 (Disease Network, Target Analysis, Cancer Analysis)
- [ ] Implement Scenarios 4-6 (MRA Simulation, Pathway Comparison, Drug Repurposing)
- [ ] Implement integration tests for all 6 scenarios
- [ ] Implement caching, parallel processing, and memory optimization
- [ ] Implement benchmark validation suite with DREAM/TCGA/COSMIC datasets
- [ ] Complete API reference, user guide, and scientific background documentation