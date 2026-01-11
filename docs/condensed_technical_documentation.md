# OmniTarget Technical Documentation

## Repository Guidelines

### Project Structure & Module Organization
The pipeline is Python-first: `src/` hosts the CLI (`cli.py`, `cli/`), orchestration layers (`core/`, `mcp_clients/`), scenarios (`scenarios/`), and supporting utilities/visualization. Configs stay in `config/` (especially `mcp_servers.json`), YAML examples under `examples/yaml_configs/`, datasets in `data/`, and generated artifacts in `results/` or `logs/`. Prefer the suites in `tests/` (`unit`, `fast`, `integration`, `performance`, `production`, `benchmark`, `fixtures/`); legacy scenario-specific tests at the root exist only for backward compatibility.

### Build, Test, and Development Commands
- `python -m pip install -e .[dev]` — dev install including lint/type extras from `setup.py`.
- `python -m src.cli yaml examples/yaml_configs/axl_breast_cancer.yaml` — batch workflow run.
- `python run_efficient_tests.py --mode comprehensive` — curated comprehensive suite for core functionality.
- `python run_efficient_tests.py --mode dev --parallel` — fast/unit/optimization pytest bundles defined in `run_efficient_tests.py`.
- `python run_benchmark_tests.py --categories dream tcga` — executes the benchmark validations in `tests/benchmark/`.
- `python start_mcp_servers.py` — boots and health-checks KEGG/Reactome/STRING/HPA servers before `pytest -m production`.

### Coding Style & Naming Conventions
Use PEP 8 (4 spaces), type hints, and docstrings on scenario entry points. Keep modules snake_case (`scenario_4_mra_simulation.py`), classes PascalCase (`MCPClientManager`), and CLI commands aligned with the `omnitarget` entry point. Run `black`, `isort`, `flake8`, and `mypy src --ignore-missing-imports` before pushing; the same checks run in `.github/workflows/test.yml`.

### Testing Guidelines
Pytest discovers everything under `tests/` per `pytest.ini` (`testpaths = tests`, `test_*.py`), so keep naming consistent and reuse the published markers (`unit`, `fast`, `integration`, `performance`, `production`, `benchmark`, `optimization`, etc.). Target ≥85% coverage with `pytest --cov=src --cov-report=html`, mirroring `P0_FIXES_TODO_LIST.md` and the Codecov upload. Run `python run_efficient_tests.py --mode integration` before scenario changes, reserve `pytest tests/production -m production` or `python run_production_tests.py` for live MCP servers, and schedule the 10–60 minute benchmark/performance suites so they do not block CI.

### Commit & Pull Request Guidelines
Upstream history follows Conventional Commits (`type(scope): summary`), e.g., `fix(scenario5): guard completeness metrics when HPA stalls`; keep bodies short, imperative, and scoped to the scenario or MCP client you touched. Pull requests should summarize the change, note affected modules/scenarios, link any P0/P1 planning docs or issues, attach artifacts (logs under `logs/`, JSON under `results/`), and only request review once the workflows in `.github/workflows/test.yml` pass locally.

### Security & Configuration Tips
Never store credentials in-source: `config/mcp_servers.json` should hold only executable paths, with secrets injected via environment variables or CI secrets. GitHub Actions expects `SYNAPSE_*` and `DRUGBANK_*` values for benchmark downloads—keep them in repository settings. Scrub sensitive data from `results/`/`logs/` before committing and shut down `start_mcp_servers.py` when ports no longer need be exposed.

## Comprehensive Fixes Documentation

### Critical Fixes Applied

#### 1. **S2 Drug Data Parsing Fix** (`scenario_2_target_analysis.py`)

**Problem:**
- `AttributeError: 'str' object has no attribute 'get'` in `_phase5_druggability_assessment`
- KEGG `search_drugs` sometimes returns string drug IDs instead of dictionaries
- Code was calling `.get()` on string objects

**Solution:**
Added defensive parsing to handle both string and dict formats:

```python
# Defensive parsing - handle dict/string formats
if isinstance(drug_data, str):
    # Drug ID string, construct dict with target_id
    drug_info = {
        'drug_id': drug_data,
        'target_id': target.gene_symbol,
        'interaction_type': 'unknown',
        'confidence': 0.5
    }
elif isinstance(drug_data, dict):
    # Already a dict, ensure it has required fields
    drug_info = drug_data.copy()
    if 'drug_id' not in drug_info and 'id' in drug_info:
        drug_info['drug_id'] = drug_info['id']
    if 'target_id' not in drug_info:
        drug_info['target_id'] = target.gene_symbol
```

#### 2. **S2 DrugInfo Type Mismatch Fix** (`scenario_2_target_analysis.py`)

**Problem:**
- `pydantic_core._pydantic_core.ValidationError: 1 validation error for TargetAnalysisResult`
- `known_drugs.0: Input should be a valid dictionary or instance of DrugInfo`
- `TargetAnalysisResult.known_drugs` expects `DrugInfo` objects, but we were passing `DrugTarget` objects

**Solution:**
Added conversion logic from `DrugTarget` → `DrugInfo`:

```python
# Convert DrugTarget objects to DrugInfo objects for known_drugs
drug_targets = druggability_data.get('drugs', []) if isinstance(druggability_data, dict) else []
known_drugs = []
for drug_target in drug_targets:
    if isinstance(drug_target, DrugTarget):
        # Convert DrugTarget to DrugInfo
        drug_info = DrugInfo(
            drug_id=drug_target.drug_id,
            name=drug_target.drug_id,  # Use drug_id as name if not available
            indication=None,
            mechanism=drug_target.mechanism,
            targets=[drug_target.target_id],
            development_status=None,
            drug_class=None,
            approval_status=None
        )
        known_drugs.append(drug_info)
    elif isinstance(drug_target, DrugInfo):
        # Already a DrugInfo, use as-is
        known_drugs.append(drug_target)
    elif isinstance(drug_target, dict):
        # Handle dict format (DrugTarget or DrugInfo)
        # ... conversion logic ...
```

#### 3. **HPA Expression Parsing Fix (S1, S2, S4)**

**Problem:**
- `'list' object has no attribute 'get'` errors throughout pipeline
- HPA `get_tissue_expression` returns inconsistent formats (lists vs dictionaries)
- Multiple scenarios manually parsing HPA expression data with redundant code

**Solution:**
Standardized all scenarios to use HPA parsing helpers:

**Helper Functions** (`src/utils/hpa_parsing.py`):
- `_iter_expr_items()`: Handles both list and dict formats, extracts (tissue, nTPM) pairs
- `categorize_expression()`: Converts nTPM values to categorical levels ('High', 'Medium', 'Low', 'Not detected')

**Updated Scenarios:**
- **S1** (`scenario_1_disease_network.py`): Replaced manual parsing with `_iter_expr_items()` and `categorize_expression()` (lines 962-987)
- **S2** (`scenario_2_target_analysis.py`): Replaced manual parsing (lines 506-527)
- **S4** (`scenario_4_mra_simulation.py`): Replaced manual parsing (lines 344-361)

## MCP Concurrency Fix

### Problem Description

**Error:**
```
MCPServerError: ... RuntimeError: readuntil() called while another coroutine is already waiting for incoming data
```

**Root Cause:**
- Multiple concurrent requests to the **same MCP server** (KEGG, Reactome, STRING)
- Node.js MCP servers can't handle concurrent stdio access
- Multiple async tasks reading from the same stdout simultaneously
- Causes `readuntil()` conflict in the Node.js JSON-RPC library

### Solution Implemented

#### Per-Server Semaphore

Added `asyncio.Semaphore(1)` to each MCP client instance:

```python
# In MCPSubprocessClient.__init__()
self._server_semaphore = asyncio.Semaphore(1)

# In MCPSubprocessClient.call_tool()
async with self._server_semaphore:
    # Serialize all stdio operations (read/write) for this server
```

**How It Works:**
1. **Same Server**: Serializes requests to one at a time
2. **Different Servers**: Allows parallel execution
3. **Maintains Performance**: Still get parallelism across different databases

**Benefits:**
- ✅ Fixes "readuntil()" errors
- ✅ Maintains P0-3 performance improvements
- ✅ Different servers run in parallel
- ✅ No impact on batch query framework
