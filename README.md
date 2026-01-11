# OmniTarget: Target Discovery & Network Analysis

OmniTarget is a Python pipeline for network-based target discovery and pathway analysis. It orchestrates multiple Model Context Protocol (MCP) servers (KEGG, Reactome, STRING, HPA, UniProt, ChEMBL) to build disease networks, characterize targets, and support network pharmacology workflows.

## Core capabilities
- Multi-database disease and pathway discovery
- Protein interaction networks and expression overlays
- Cancer-specific biomarker and tissue expression analysis
- Modular Response Analysis (MRA) simulations
- Cross-pathway enrichment and comparison
- Network-based drug repurposing and off-target analysis

## Scenarios
1. Disease network construction
2. Target analysis
3. Cancer analysis
4. MRA simulation
5. Pathway comparison
6. Drug repurposing

## Quick start

### 1) Install
```bash
python -m pip install -e .
```

### 2) Configure MCP servers
Copy the example and point each server to your local MCP builds:
```bash
cp config/mcp_servers.example.json config/mcp_servers.json
```
Edit `config/mcp_servers.json` to match your MCP server paths.

Optional environment config:
```bash
cp config/env.example .env
```

### 3) Run a basic health check
```bash
omnitarget health
```

### 4) Run a sample analysis
```bash
python -m src.cli yaml examples/yaml_configs/axl_breast_cancer.yaml
```

### 5) Visualize results
```bash
python -m src.cli visualize results/axl_breast_cancer_analysis.json
```

## CLI usage
```bash
omnitarget example
omnitarget health
python -m src.cli interactive
python -m src.cli yaml examples/yaml_configs/axl_breast_cancer.yaml
python -m src.cli visualize results/analysis.json --interactive --format all
```

## Testing
Recommended suite (comprehensive):
```bash
python run_efficient_tests.py --mode comprehensive
```

Additional modes:
```bash
python run_efficient_tests.py --mode dev
python run_efficient_tests.py --mode integration
python run_efficient_tests.py --mode performance
```

See `tests/README.md` for suite details and guidance.

## Outputs
- Default output directory: `results/`
- Development timeline (logs, run artifacts): `docs/development-timeline/`

## Documentation
- Documentation index: `docs/README.md`
- Technical overview: `condensed_technical_documentation.md`
- Progress/testing summary: `condensed_progress_testing.md`
- Results analysis: `condensed_results_analysis.md`

## Repository structure
```
config/            MCP server config and runtime settings
src/               Core pipeline and scenario implementations
tests/             Unit, integration, performance, production tests
examples/          Example scripts and YAML configs
docs/              Documentation and development timeline
archive/           Legacy tests and historical artifacts
```

## Related Projects
- [netpharm-viz](https://github.com/OASolliman590/netpharm-viz) - Standalone network visualization toolkit (proof-of-concept)

## GitHub
Repository: https://github.com/OASolliman590/omnitarget
Issues: https://github.com/OASolliman590/omnitarget/issues
