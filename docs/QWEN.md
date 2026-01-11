# OmniTarget Bioinformatics Pipeline - Development Context

## Project Overview

OmniTarget is a sophisticated bioinformatics workflow integration platform that orchestrates multiple biological databases through Model Context Protocol (MCP) servers. The pipeline performs comprehensive multi-scenario analysis of protein targets in disease contexts, with a focus on cancer research. The system integrates data from KEGG, Reactome, STRING, Human Protein Atlas (HPA), UniProt, and ChEMBL databases to perform systematic pathway analysis, target characterization, and drug repurposing studies.

The project primarily focuses on analyzing the AXL receptor tyrosine kinase in breast cancer, though the framework is generalizable to other targets and diseases. It executes six distinct analytical scenarios through an integrated pipeline architecture with comprehensive validation and optimization components.

## Core Architecture

### Main Components
- **MCP Client Manager**: Centralized manager for all biological database clients with lifecycle management
- **Six Analytical Scenarios**: Modular analysis workflows for different research questions
- **Optimization Components**: Caching, parallel processing, memory optimization, and connection pooling
- **Visualization Pipeline**: Network visualization and result presentation
- **Validation Framework**: Comprehensive validation and scoring system

### Analytical Scenarios
1. **Disease Network Analysis**: Multi-database disease discovery and pathway mapping
2. **Target Characterization**: Network topology, expression profiling, and druggability assessment
3. **Cancer-Specific Analysis**: Tumor-specific gene networks and therapeutic targets
4. **Multi-Reaction Analysis (MRA)**: Network perturbation simulations
5. **Pathway Comparison**: Cross-database pathway validation
6. **Drug Repurposing**: Multi-pathway targeting strategies

### Technologies & Dependencies
- Python 3.8+
- NetworkX for network analysis
- AsyncIO for concurrent database queries
- Pydantic for data validation
- Various visualization libraries (Matplotlib, Plotly, PyVis)
- MCP (Model Context Protocol) servers for database access

## Database Integration

The system connects to multiple biological databases through dedicated MCP servers:
- **KEGG**: Pathway and disease information
- **Reactome**: Detailed pathway interactions
- **STRING**: Protein-protein interaction networks
- **HPA**: Tissue expression profiles
- **UniProt**: Protein annotations and disease associations
- **ChEMBL**: Drug and compound information

## Building and Running

### Prerequisites
- Python 3.8 or higher
- Node.js (for MCP servers)
- Access to configured MCP servers
- Redis (for caching - optional)

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Run basic health check
omnitarget health

# Run example
omnitarget example

# Run curated comprehensive suite
python run_efficient_tests.py --mode comprehensive
```

### Main Execution
The pipeline can be executed through the CLI or YAML runner:

```bash
# Run the comprehensive AXL breast cancer analysis
python -m src.cli yaml examples/yaml_configs/axl_breast_cancer.yaml
```

### Configuration
Main configuration is stored in `config/mcp_servers.json` which specifies the paths to each database's MCP server.

## Development Conventions

### Code Structure
- `src/`: Main source code organized by functionality
  - `cli/`: Command-line interface
  - `core/`: Core pipeline functionality
  - `mcp_clients/`: Database client implementations
  - `models/`: Data models using Pydantic
  - `scenarios/`: Six analytical scenario implementations
  - `utils/`: Utility functions
  - `visualization/`: Network visualization components

### Error Handling
- Comprehensive exception handling with specific error types
- Database connection errors, timeouts, and validation errors
- Graceful fallback mechanisms
- Detailed error logging and reporting

### Testing Approach
- Multiple test categories: unit, integration, performance, production
- Real-world biological test cases (e.g., AXL in breast cancer)
- Comprehensive validation of results against literature
- Benchmark testing and optimization validation

## Key Features

1. **Multi-Source Data Integration**: Combines pathway, network, expression, and drug data
2. **Six-Scenario Analysis**: Comprehensive analytical approach to target validation
3. **Performance Optimization**: Caching, parallel processing, memory management
4. **Validation Framework**: Scoring system and cross-database validation
5. **Network Simulation**: MRA modeling for perturbation analysis
6. **Visualization Support**: Network graphs and interactive visualizations
7. **Modular Architecture**: Pluggable components and scenario-based design

## Project Status

The project is in an advanced development stage with comprehensive documentation of results, extensive testing, and implemented fixes for various issues including:
- Database connection problems
- Reactome pathway extraction improvements
- Concurrency and performance optimizations
- Memory management enhancements
- Error handling improvements
- Production readiness features

The pipeline has demonstrated successful execution across all six scenarios with validation scores ranging from 0.13 to 0.95 and data completeness scores between 50.0% and 88.8%.
